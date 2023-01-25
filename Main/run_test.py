import os, sys, argparse


def set_path():
    """
    * 경로 잡기
    """

    # 경로명 설정
    AppPath = os.path.dirname(os.path.abspath(os.getcwd()))
    SRC_DIR_NAME_Common = 'Common'
    SRC_DIR_NAME_DATA = 'DATA'
    SRC_DIR_NAME_DeepLearning = 'DeepLearning'
    SRC_DIR_NAME_Lib = 'Lib'
    SRC_DIR_NAME_LOG = 'LOG'
    SRC_DIR_NAME_Main = 'Main'
    SRC_DIR_NAME_RES = 'RES'

    Common = os.path.join(AppPath, SRC_DIR_NAME_Common)
    DATA = os.path.join(AppPath, SRC_DIR_NAME_DATA)
    DeepLearning = os.path.join(AppPath, SRC_DIR_NAME_DeepLearning)
    Lib = os.path.join(AppPath, SRC_DIR_NAME_Lib)
    LOG = os.path.join(AppPath, SRC_DIR_NAME_LOG)
    Main = os.path.join(AppPath, SRC_DIR_NAME_Main)
    RES = os.path.join(AppPath, SRC_DIR_NAME_RES)

    # 경로 추가
    AppPathList = [AppPath, Common, DATA, DeepLearning, Lib, LOG, Main, RES]
    for p in AppPathList:
        sys.path.append(p)


def arguments():
    """
    * parser 이용하여 프로그램 실행 인자 받기
    :return: args
    """

    from Common import ConstVar

    # parser 생성
    parser = argparse.ArgumentParser(prog="Deep Learning Study Project Test",
                                     description="* Run this to test the model.")

    # parser 인자 목록 생성
    # 테스트 데이터 디렉터리 설정
    parser.add_argument("--test_data_dir",
                        type=str,
                        help='set test data directory',
                        default=ConstVar.DATA_DIR_TEST,
                        dest="test_data_dir")

    # 불러올 체크포인트 파일 경로
    parser.add_argument("--checkpoint_file",
                        type=str,
                        help='set checkpoint file to load if exists',
                        default=None,
                        dest='checkpoint_file')

    # parsing 한거 가져오기
    args = parser.parse_args()

    return args


def run_program(args):
    """
    * 테스트 실행
    :param args: 프로그램 실행 인자
    :return: None
    """

    import torch
    from torch.utils.data import DataLoader

    from Common import ConstVar
    from DeepLearning.test import Tester
    from DeepLearning.dataloader import SIGNSDataset
    from DeepLearning.model import Generator, Discriminator
    from DeepLearning.metric import bce_loss

    # GPU / CPU 설정
    device = ConstVar.DEVICE_CUDA if torch.cuda.is_available() else ConstVar.DEVICE_CPU

    # 모델 선언
    modelG = Generator()
    modelD = Discriminator()
    # 각 모델을 해당 디바이스로 이동
    modelG.to(device)
    modelD.to(device)

    # 테스트용 데이터로더 선언
    test_dataloader = DataLoader(dataset=SIGNSDataset(data_dir=args.test_data_dir,
                                                      mode_train_test=ConstVar.MODE_TEST))

    # 모델 테스트 객체 선언
    tester = Tester(modelG=modelG,
                    modelD=modelD,
                    metric_fn=bce_loss,
                    test_dataloader=test_dataloader,
                    device=device)

    # 모델 테스트
    tester.running(checkpoint_file=args.checkpoint_file)


def main():

    # 경로 잡기
    set_path()

    # 인자 받기
    args = arguments()

    # 프로그램 실행
    run_program(args=args)


if __name__ == '__main__':
    main()
