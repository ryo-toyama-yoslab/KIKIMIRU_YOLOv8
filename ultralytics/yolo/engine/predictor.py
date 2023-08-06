# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlmodel            # CoreML (macOS-only)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
"""
import platform
from pathlib import Path

import cv2
import os
import numpy as np
import torch
import yaml
import shutil
import time
import datetime
import re
import sys

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.data.augment import LetterBox, classify_transforms
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, MACOS, SETTINGS, WINDOWS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = None
        if self.args.conf is None:
            self.args.conf = 0.7  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def yaml_save(self, file='data.yaml', data=None):
        if data is None:
            data = {}
        file = Path(file)
        if not file.parent.exists():
            # Create parent directories if they don't exist
            file.parent.mkdir(parents=True, exist_ok=True)

        # Convert Path objects to strings
        for k, v in data.items():
            if isinstance(v, Path):
                data[k] = str(v)

        # Dump data to file in YAML format
        with open(file, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    
    def setting_dir_yaml_load(file="", append_filename=False):
        file = file = Path().home() / "yolov8_config/Predict/save_dir.yaml"
        
        with open(file, errors='ignore', encoding='utf-8') as f:
            s = f.read()  # string

            # Remove special characters
            if not s.isprintable():
                s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

            # Add YAML filename to dict and return
            return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)

    def get_save_dir(self):
        root_home = Path().home()
        project = root_home / Path(SETTINGS['runs_dir'])
        '''
        "yolov8_config/Predict/save_dir.yaml"で保存されている出力先フォルダを読み込み
        '''
        save_file_setting = self.setting_dir_yaml_load() # 現在時間を名前にしたフォルダ名を読み込む
        
        return increment_path(Path(project) / save_file_setting['time'], exist_ok=self.args.exist_ok)
    
    def setting_save_dir(self):

        # アプリが開始されたことを検知した時間で作られるフォルダパスを読み書きするファイル
        file = Path().home() / "yolov8_config/Predict/save_dir.yaml"

        dt_now = datetime.datetime.now()
        
        defaults = {
            'time': dt_now.strftime('%Y%m%d%H%M%S') # now time
            }
        
        self.yaml_save(file, defaults)
    

    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def inference(self, im, *args, **kwargs):
        visualize = increment_path(self.save_dir / Path(self.batch[0][0]).stem,
                                   mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
        return self.model(im, augment=self.args.augment, visualize=visualize)

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes and self.model.pt
        return [LetterBox(self.imgsz, auto=auto, stride=self.model.stride)(image=x) for x in im]

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)
        return log_string

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode."""
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs
    
    def write_postdata(self, path, post_data):
        """
        Write data for post specify result to application
        """
        try:
            with open(path, 'w') as f:
                f.write(post_data)
                os.chmod(path, 0o644)
        except Exception as e:
            LOGGER.info(f"Error in Writing Post result : {e}")

    def reset_postdata(self, path):
        """
        Reset data for post post specify result to application
        """
        try:
            with open(path, 'w') as f:
                f.write("")
                os.chmod(path, 0o644)
        except Exception as e:
            LOGGER.info(f"Error in Writing Post result : {e}")

    class MyHandler(FileSystemEventHandler):

        def __init__(self, observer):
            self.observer = observer

        def on_created(self, event):
            if event.is_directory:
                return
            self.observer.stop()

    def watch_folder(self, folder_path):
        observer = Observer()
        event_handler = self.MyHandler(observer)
        observer.schedule(event_handler, path=folder_path, recursive=False)
        observer.start()
        try:
            while observer.is_alive():
                time.sleep(0.01) # 監視処理の間隔を0.01秒にして負荷軽減
        except KeyboardInterrupt:
            sys.exit()
    
    # Specific medical practice
    def specificResult(self, results_list): # ラベル番号と認識数の辞書型リストが引数
        """
        results_listのインデックスとラベル名の対応
        0 : 'blood_cl_bottle_blue'
        1 : 'blood_cl_bottle_orange'
        2 : 'central_venous_catheter'
        3 : 'guide_wire'
        4 : 'mark_needle'
        5 : 'spinal_needle'
        """
        
        # 特定できないときはunknown
        post_result = "unknown" 

        # 各医療機器認識割合を格納するリスト
        results_rate = [0] * 6

        # Start specific medical practice
        # mark_num = results.count('mark_needle')
        # spinal_num = results.count('spinal_needle')
        # catheter_num = results.count('central_venous_catheter')
        # wire_num = results.count('guide_wire')
        # bottle_orange_num = results.count('blood_cl_bottle_orange')
        # bottle_blue_num = results.count('blood_cl_bottle_blue')
        
        # results_sum = mark_num + spinal_num + catheter_num + wire_num + bottle_orange_num + bottle_blue_num
        results_sum = sum(results_list)

        try:
            for i in range(len(results_list)):
                results_rate[i] = results_list[i] / results_sum
        except ZeroDivisionError:
            return post_result

        # 認識割合を降順ソート，インデックスを格納
        sortedRate_indices = sorted(range(len(results_rate)), key=lambda i: results_rate[i], reverse=True)
        
        # bottle_blue_rate = results_list[0] /results_sum
        # bottle_orange_rate = results_list[1] / results_sum
        # catheter_rate = results_list[2] / results_sum
        # gwire_rate = results_list[3] / results_sum
        # mark_rate = results_list[4] / results_sum
        # spinal_rate = results_list[5] / results_sum

        #print(f"result_rate  mark_rate:{mark_rate} spinal_rate:{spinal_rate} catheter_rate:{catheter_rate} gwire_rate:{gwire_rate} bottle_orange_rate:{bottle_orange_rate} bottle_blue_rate:{bottle_blue_rate}")
        
        
        
        # 認識割合を降順にソート
        # results_rate = sorted(results_rate.items(), key=lambda x: x[1], reverse=True)
        
        print(f"results_list : {results_list}")
        print(f"results_sum : {results_sum}")
        print(f"results_rate : {results_rate}")
        print(f"sortedRate_indices : {sortedRate_indices}")

        top_rate_results = [sortedRate_indices[0]] # 認識割合トップの医療機器ラベル番号(複数抽出有)
        for i in range(len(sortedRate_indices)-1):
            idx_top = sortedRate_indices[0]
            idx_temp = sortedRate_indices[i+1]
            if results_rate[idx_temp] == results_rate[idx_top]:
                top_rate_results.append(idx_temp)
            else:
                break
        print(f"top_rate_results : {top_rate_results}")
        # 1種類の医療機器だけが認識率トップの場合の医療行為特定
        if len(top_rate_results) == 1:
            # 認識割合2位の医療機器(複数抽出有)
            second_rate_results = [sortedRate_indices[1]] 
            for j in range(len(sortedRate_indices)-2):
                idx_second = sortedRate_indices[1]
                idx_second_temp = sortedRate_indices[j+2]
                if results_rate[idx_second_temp] == results_rate[idx_second]:
                    second_rate_results.append(idx_second_temp)
                else:
                    break
            print(f"second_rate_results : {second_rate_results}")
            
            # オレンジと青のボトルのどちらかが認識率トップかつもう片方が2番目に認識数が多い場合は血液培養と判断
            if top_rate_results[0] in [0, 1]:
                print(f"top_results in [0, 1] : {top_rate_results}")
                if len(second_rate_results) == 1 and second_rate_results[0] in [0, 1]:
                    print(f"second_results in [0, 1] : {second_rate_results[0]}")
                    return "blood"
            # カテーテルとガイドワイヤーのどちらかが認識率トップかつもう片方が2番目に認識数が多い場合はカテーテルと判断
            elif top_rate_results[0] in [2, 3]:
                print(f"top_results in [2, 3] : {top_rate_results}")
                if len(second_rate_results) == 1 and second_rate_results[0] in [2, 3]: 
                    print(f"scond_results in [2, 3] : {second_rate_results}")
                    return "catheter"
            elif top_rate_results[0] == 4: # 骨髄穿刺と判断
                return "kotuzui"
            elif top_rate_results[0] == 5: # 腰椎穿刺と判断
                return "youtui"
        
        
        # 認識割合トップが2つの場合に特定できる医療行為(3つ以上は現状特定できる医療行為無し)
        if len(top_rate_results) == 2:
            if set(top_rate_results) == {0, 1}:
                return "blood"
            elif set(top_rate_results) == {2, 3}:
                return  "catheter"
        
        print(f"post_result : {post_result}")
        return post_result
            

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)

        '''この場所に処理を追加
        
        1. 認識対象画像が保存されるフォルダに画像が来たかを監視する処理

        2. 画像保存を確認したらその時間を元に出力先のフォルダを作成
             - save_dir.yamlに出力先のフォルダ名を書き込み(setting_save_dir関数)
               get_save_dir関数でそのyamlファイルを読み込んで出力先を指定
           アプリが実行中だと考えられる間は画像フォルダに保存される画像を読み込み続ける
             - 一定時間画像が保存されなかったらアプリが実行されていないと判断して処理を終了する

        3. 再びフォルダに画像が保存されるかを監視する処理に戻る

        '''

        while True:# 認識対象画像フォルダの変更を監視 新規画像が入ってきたら認識ループに入る
            folder_path = source if source is not None else self.args.source
            post_result_path = Path.home() / 'public_html' / 'kikimiru_server' / 'post_result' / 'post_result.txt'
            print("start watch_folder")
            self.watch_folder(folder_path)
            print("end watch_folder")

            # setting save_dir
            self.setting_save_dir()

            # setup save_dir from yaml file
            self.save_dir = self.get_save_dir()

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
                (self.save_dir / 'predicted_image' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            #----------------------- 認識ループ開始位置 -----------------------#
            predict_flag = True
            predicted_label_log = [0] * 6  # 予測結果ラベル(6種類)を記録しておく辞書
            while predict_flag:
                print("start predict")
                time.sleep(0.01) # 0.01秒の待機時間を設定(画像フォルダに画像が保存されるまで待機)

                # Setup source every time predict is called - 認識対象画像データ読み込み
                self.setup_source(source if source is not None else self.args.source)

                # Warmup model 
                if not self.done_warmup:
                    self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                    self.done_warmup = True

                self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
                self.run_callbacks('on_predict_start')
                for batch in self.dataset:
                    self.run_callbacks('on_predict_batch_start')
                    self.batch = batch
                    path, im0s, vid_cap, s = batch

                    # Preprocess
                    with profilers[0]:
                        im = self.preprocess(im0s)

                    # Inference
                    with profilers[1]:
                        preds = self.inference(im, *args, **kwargs)

                    # Postprocess
                    with profilers[2]:
                        self.results = self.postprocess(preds, im, im0s)
                    self.run_callbacks('on_predict_postprocess_end')

                    # Visualize, save, write results
                    n = len(im0s)
                    for i in range(n):
                        self.seen += 1
                        self.results[i].speed = {
                            'preprocess': profilers[0].dt * 1E3 / n,
                            'inference': profilers[1].dt * 1E3 / n,
                            'postprocess': profilers[2].dt * 1E3 / n}
                        p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                        p = Path(p)
                        '''
                        p : 認識対象画像のファイルパス
                        im : 元画像の正規化済み画素値行列
                        im0 : 元画像の画素値行列
                        '''
                        if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                            s += self.write_results(i, self.results, (p, im, im0))
                            label_list = self.results[i].get_prdLabel_list() # 認識結果のラベル番号をリストで取得 1枚に複数結果なら複数の番号
                            if len(label_list) > 0: # 認識結果がある場合
                                for j in range(len(label_list)):
                                    predicted_label_log[label_list[j]] = predicted_label_log[label_list[j]] + 1
                                # 認識結果から医療行為特定
                                post_result = self.specificResult(predicted_label_log)
                                print(f"post_result : {post_result}")
                                self.write_postdata(post_result_path, post_result)
                            print(f"predicted_label_log : {predicted_label_log}")
                        if self.args.save or self.args.save_txt:
                            self.results[i].save_dir = self.save_dir.__str__()
                        if self.args.show and self.plotted_img is not None:
                            self.show(p)
                        if self.args.save and self.plotted_img is not None:
                            self.save_preds(vid_cap, i, str(self.save_dir / p.name))
                    
                    self.run_callbacks('on_predict_batch_end')
                    yield from self.results

                    # Print time (inference-only)
                    if self.args.verbose:
                        LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')
                    
                    # 認識済み画像kikimiru_detection/yolov8_results/{推論開始時間}/predicted_imageフォルダ
                    file_name = os.path.basename(p)
                    path_to_move = self.save_dir / 'predicted_image' /  file_name

                    try:
                        # 画像をコピー
                        if shutil.copy(p, path_to_move):
                            os.chmod(str(path_to_move),0o740)
                            # 元画像を削除
                            os.remove(p)
                        else:
                            print(f"Failed to copy original image : {p}")
                        # shutil.move(str(p), str(path_to_move))
                    except FileNotFoundError:
                        print("File not found from which to move")
                    except PermissionError:
                        print("The source or destination file does not have access permissions")
                    except shutil.Error as e:
                        print(f"Failed to move file : {e}")
                
                # 認識対象画像フォルダに画像があるかを確認
                files_in_folder = os.listdir(folder_path)
                if len(files_in_folder) == 0:
                    time_predict_start = time.time()
                    while True:
                        time.sleep(0.001) # 0.001秒の待機時間を設定(画像があるかを確認する間隔)
                        files_in_folder = os.listdir(folder_path)
                        if len(files_in_folder) > 0:
                            break
                        if time.time() - time_predict_start > 300: # 5秒間入力画像がなければ認識状態から画像待機状態に移行
                            # Release assets
                            if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                                self.vid_writer[-1].release()  # release final video writer
                            # Print results
                            if self.args.verbose and self.seen:
                                t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
                                LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                                            f'{(1, 3, *im.shape[2:])}' % t)
                            if self.args.save or self.args.save_txt or self.args.save_crop:
                                nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
                                s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
                                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
                            self.run_callbacks('on_predict_end')

                            self.reset_postdata(post_result_path) # postデータと画像保存フォルダのリセット
                            predicted_label_log = [0] * 6 # 認識結果を初期化
                            predict_flag = False # 認識状態をFalseに
                            break
            #----------------------- 認識ループ終了位置 -----------------------#
        

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(model or self.args.model,
                                 device=select_device(self.args.device, verbose=verbose),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        im0 = self.plotted_img
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        # Save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
            os.chmod(save_path,0o740)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix = '.mp4' if MACOS else '.avi' if WINDOWS else '.avi'
                fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'MJPG'
                save_path = str(Path(save_path).with_suffix(suffix))
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """
        Add callback
        """
        self.callbacks[event].append(func)
