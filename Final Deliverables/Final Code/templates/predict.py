import os
import re
import queue
import shutil
import time
import threading
import json
import time
from imageai.Prediction.Custom import CustomImagePrediction
from logger import Logger


class TestPredictionThread:
    '''
    ImageAI Custom Prediction功能物件，針對鏡面汙染圖檔進行預測分類與報告產出
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    __execution_path = os.getcwd()
    __logger = Logger().get_logger()

    def __init__(self):
        self.__thread_num = 3
        # self.__test_folder_path = None
        self.__test_images_queue = None
        self.__test_model_path = None
        self.__test_json_path = None
        self.__prediction_result_path = None
        self.__prediction_result = []

    def __custom_split(self, separators, str_to_split):
        if isinstance(str_to_split, str):
            # create regular expression dynamically
            regular_exp = '|'.join(map(re.escape, separators))
            return re.split(regular_exp, str_to_split)
        else:
            raise ValueError('Please check your string content!')

    def set_test_info(self, test_info: dict):
        if isinstance(test_info, dict):
            # self.__test_folder_path = test_info["folder_path"]
            self.__test_model_path = test_info["model_path"]
            self.__test_json_path = test_info["json_path"]
            # self.__prediction_result_path = self.create_prediction_result_path(
            #     test_info["prediction_path"])
        else:
            raise ValueError('Please check your test info!')

    def set_prediction(
        self,
        model_type: str = "ResNet",
        thread_num: str = "1",
    ) -> CustomImagePrediction:
        '''設定Custom Prediction AI Model
           請注意，model type必須要與training用的model type相同

        Args:
            model_type: 欲使用之model的型態，預設為ResNet，可選擇SqueezeNet, InceptionV3, DenseNet 
            thread_num: 要開的thread數量

        Returns:
            prediction: custom prediction模組，裡面包含設定的 setModelType、setModelPath、setJsonPath、loadModel
        '''
        try:
            if isinstance(model_type, str):
                model_path = self.__test_model_path
                json_path = self.__test_json_path
                prediction_model = self.get_predection_model(model_path)
                prediction_json = self.get_json_file_path(json_path)
                model_class_info = self.get_model_class_info(prediction_json)

                prediction = CustomImagePrediction()
                if model_type == "SqueezeNet":
                    prediction.setModelTypeAsSqueezeNet()
                elif model_type == "ResNet":
                    prediction.setModelTypeAsResNet()
                elif model_type == "InceptionV3":
                    prediction.setModelTypeAsInceptionV3()
                elif model_type == "DenseNet":
                    prediction.setModelTypeAsDenseNet()
                else:
                    raise ValueError("Please check your model!!")
                prediction.setModelPath(prediction_model)
                prediction.setJsonPath(prediction_json)
                prediction.loadModel(num_objects=int(len(model_class_info)))
                self.__logger.info(
                    f"Thread[{thread_num}], Model path:{prediction_model}, Model type: {model_type}"
                )
                return prediction

            else:
                raise ValueError("Please check your folder path!")
        except Exception as e:
            self.__logger.error(e)

    def get_all_images(
        self,
        image_path: str,
    ) -> queue:
        '''取得所有測試圖片

        Args:
            folder_path: 放test的檔案夾位置

        Returns:
            image_queue: 存放所有image路徑的queue
        '''
        try:
            if isinstance(image_path, str):
                if os.path.exists(image_path):
                    image_queue = queue.Queue()
                    if image_path.endswith(('.png', '.jpg', 'jpeg', 'PNG', 'JPG', 'JPEG')):
                        image_queue.put(image_path)
                    elif os.path.isdir(image_path):
                        for root, dirs, files in os.walk(image_path):
                            for f in files:
                                if f.endswith(('.png', '.jpg', 'jpeg', 'PNG', 'JPG', 'JPEG')):
                                    fullpath = os.path.join(root, f)
                                    image_queue.put(fullpath)
                    else:
                        raise ValueError('Content is not a image or folder!')
                    return image_queue
                else:
                    raise IOError("Can't find path content!")
            else:
                raise ValueError('Please check your folder path!')
        except IOError as e:
            self.__logger.error(e)

    def get_all_folders(
        self,
        folder_path: str,
    ) -> list:
        '''取得所有測試圖片的資料夾路徑

        Args:
            folder_path: 放test的檔案夾位置

        Returns:
            存放所有folder的名稱的list
        '''
        try:
            if isinstance(folder_path, str):
                if os.path.exists(folder_path):
                    folder = []
                    for root, dirs, files in os.walk(folder_path):
                        for dir in dirs:
                            folder.append(dir)
                    self.__logger.info(f"all test folder: {folder}")
                    return folder
                else:
                    raise IOError("Can't find folder from path!")
            else:
                raise ValueError('Please check your folder path!')
        except IOError as e:
            self.__logger.error(e)

    def create_prediction_result_path(self, result_path: str) -> str:
        '''創建存放prediction分配照片的資料夾

        Args:
            path: 放置prediction 結果的檔案夾位置

        Returns:
            prediction_result_path: 預測結果的資料夾路徑並建立好的裡面要預測的資料夾
        '''
        try:
            if isinstance(result_path, str):

                json_path = self.__test_json_path
                prediction_model_class = self.get_json_file_path(json_path)
                prediction_model_class_info = self.get_model_class_info(
                    prediction_model_class)
                prediction_result_path = result_path
                if os.path.exists(prediction_result_path):
                    shutil.rmtree(prediction_result_path)
                    self.__logger.info(
                        f"Previous prediction folder: {prediction_result_path} has been remove!!"
                    )
                for number, item in prediction_model_class_info.items():
                    # prediction_dirs = self.get_all_folders(test_path)
                    # for dir in prediction_dirs:
                    if not os.path.exists(
                            os.path.join(prediction_result_path, item)):
                        os.makedirs(os.path.join(prediction_result_path, item))
                    for i in range(len(prediction_model_class_info)):
                        if not os.path.exists(
                                os.path.join(
                                    prediction_result_path, item,
                                    f"prediction_{prediction_model_class_info[str(i)]}"
                                )):
                            os.makedirs(
                                os.path.join(
                                    prediction_result_path, item,
                                    f"prediction_{prediction_model_class_info[str(i)]}"
                                ))
                return prediction_result_path
            else:
                raise ValueError("Please check you input path!")
        except Exception as e:
            self.__logger.error(e)

    def get_predection_model(self, folder_path: str) -> str:
        '''取得訓練出來的最高信度的model

        Args:
            folder_path: 放model的檔案夾位置

        Returns:
            top_model_apth: 最高信度model的位置
        '''
        try:
            if isinstance(folder_path, str):
                models_path = folder_path
                all_models = os.listdir(models_path)
                all_model_prediction_value = [
                    self.__custom_split("-", model)[-1] for model in all_models
                ]
                all_prediction_value = [
                    float(model_value[0:8])
                    for model_value in all_model_prediction_value
                ]
                top_model_position = all_prediction_value.index(
                    max(all_prediction_value))
                top_model_path = os.path.join(models_path,
                                              all_models[top_model_position])
                return top_model_path
            else:
                raise ValueError("Please check you input path!")
        except IOError as e:
            self.__logger.error(e)

    def get_json_file_path(
        self,
        folder_path: str,
    ) -> str:
        '''取得model class json file的位置

        Args:
            folder_path: 放置model class的位置

        Returns:
            json_path: json file的位置
        '''
        try:
            if isinstance(folder_path, str):
                jsons_path = folder_path
                all_json_files = os.listdir(jsons_path)
                json_file_path = os.path.join(jsons_path, all_json_files[0])
                return json_file_path
            else:
                raise ValueError('Please check your folder path!')
        except IOError as e:
            self.__logger.error(e)

    def get_model_class_info(self, file_path: str) -> dict:
        '''取得model class info

        Args:
            file_path: 放置model info 的json file的位置

        Returns:
            json_config: json file的資料
        '''
        if isinstance(file_path, str):
            with open(file_path, 'r') as f:
                try:
                    json_config = json.load(f)
                    f.close()
                    return json_config
                except OSError as e:
                    self.__logger.error('Invalid json: {}'.format(e))
        else:
            raise ValueError('Please check your file path!')

    def images_prediction(self,
                          test_path: str,
                          thread_num: str = "1",
                          model_type: str = "ResNet",
                          muti_thread: bool = False):
        '''取得prediction image的結果,並將預測圖片歸類到指定資料夾
        Args:
            thread_num: 要開的thread數量

        Returns: None
        '''
        try:

            prediction = self.set_prediction(model_type, thread_num)
            # prediction_result_path = self.__prediction_result_path

            # prediction_model_class = self.get_json_file_path(
            #     self.__test_json_path)
            # prediction_model_class_info = self.get_model_class_info(
            #     prediction_model_class)
            if muti_thread == False:
                self.__test_images_queue = self.get_all_images(test_path)
            while self.__test_images_queue.qsize() > 0:
                # 取得辨識圖檔的位置
                image_path = self.__test_images_queue.get()
                separators = "\\", "/"
                path_split = self.__custom_split(separators, image_path)
                file_name = path_split[-1]

                # 處理辨識資料
                prediction_info = {}
                predictions, probabilities = prediction.predictImage(
                    image_path, result_count=1)
                for eachPrediction, eachProbability in zip(
                        predictions, probabilities):
                    prediction_info["image"] = image_path
                    prediction_info["prediction"] = eachPrediction
                    prediction_info["probability"] = eachProbability
                    self.__prediction_result.append(prediction_info)
                    self.__logger.info(
                        f"Thread[{thread_num}]  Image:{image_path} ====>  Prediction: {eachPrediction} : {eachProbability}"
                    )
                    # for number, item in prediction_model_class_info.items():
                    #     if eachPrediction == item:
                    #         destination = os.path.join(
                    #             prediction_result_path, item,
                    #             f'prediction_{eachPrediction}', file_name)
                    #         shutil.copyfile(image_path, destination)

                print(
                    '--------------------------------------------------------------------------------------------------------'
                )
            return self.__prediction_result
        except Exception as e:
            self.__logger.error(e)

    # def run(self, folder_path: str):
    #     '''
    #     執行mutithreading跑custom prediction image
    #     '''
    #     thread_list = []
    #     thread_num = self.__thread_num
    #     self.__test_images_queue = self.get_all_images(folder_path)
    #     muti_thread = True

    #     for i in range(thread_num):
    #         process = threading.Thread(target=self.images_prediction,
    #                                    args=("", str(i), muti_thread),
    #                                    daemon=True)
    #         thread_list.append(process)

    #     for i in range(thread_num):
    #         thread_list[i].start()

    #     for i in range(thread_num):
    #         thread_list[i].join()

    #     return self.__prediction_result


# time_1 = time.time()
# execution_path = os.getcwd()

# model_path = os.path.join(execution_path, "idenprof", "models")
# json_path = os.path.join(execution_path, "idenprof", "json")
# prediction_path = os.path.join(execution_path, "prediction_result")
# test_info = {
#     "model_path": model_path,
#     "json_path": json_path,
#     "prediction_path": prediction_path
# }

# test_path = r"C:\Users\k09857\Desktop\idenprof\au_pollution_detect_resource\au_defect_experiment_20220502\test"
# # test_path = "L201030128_金面汙染_Total_35.png"
# # prediction_path = 'C:\\Users\\k09857\\Desktop\\idenprof\\prediction_result'
# d = TestPredictionThread()
# d.set_test_info(test_info)
# d.set_prediction()
# result = d.images_prediction(test_path)
# # result = d.run(test_path)
# print(len(result))
# time_2 = time.time()
# time_interval = time_2 - time_1
# print("time: ", time_interval)