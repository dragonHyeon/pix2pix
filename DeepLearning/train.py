import visdom
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm

from Lib import UtilLib
from Common import ConstVar
from DeepLearning import utils


class Trainer:
    def __init__(self, modelG, modelD, optimizerG, optimizerD, loss_fn, train_dataloader, device):
        """
        * 학습 관련 클래스
        :param modelG: 학습 시킬 모델. 생성자
        :param modelD: 학습 시킬 모델. 판별자
        :param optimizerG: 생성자 학습 optimizer
        :param optimizerD: 판별자 학습 optimizer
        :param loss_fn: 손실 함수
        :param train_dataloader: 학습용 데이터로더
        :param device: GPU / CPU
        """

        # 학습 시킬 모델
        self.modelG = modelG
        self.modelD = modelD
        # 학습 optimizer
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        # 손실 함수
        self.loss_fn = loss_fn
        # 학습용 데이터로더
        self.train_dataloader = train_dataloader
        # GPU / CPU
        self.device = device

    def running(self, num_epoch, output_dir, tracking_frequency, Tester, test_dataloader, metric_fn, checkpoint_file=None):
        """
        * 학습 셋팅 및 진행
        :param num_epoch: 학습 반복 횟수
        :param output_dir: 결과물 파일 저장할 디렉터리 위치
        :param tracking_frequency: 체크포인트 파일 저장 및 학습 진행 기록 빈도수
        :param Tester: 학습 성능 체크하기 위한 테스트 관련 클래스
        :param test_dataloader: 학습 성능 체크하기 위한 테스트용 데이터로더
        :param metric_fn: 학습 성능 체크하기 위한 metric
        :param checkpoint_file: 불러올 체크포인트 파일
        :return: 학습 완료 및 체크포인트 파일 생성됨
        """

        # 학습 중간 중간 생성자로 이미지를 생성하기 위한 샘플 noise z 모음
        sample_z_collection = torch.randn(size=(20, 100, 1, 1), device=self.device)

        # epoch 초기화
        start_epoch_num = ConstVar.INITIAL_START_EPOCH_NUM

        # 각 모델 가중치 초기화
        self.modelG.apply(weights_init)
        self.modelD.apply(weights_init)

        # 불러올 체크포인트 파일 있을 경우 불러오기
        if checkpoint_file:
            state = utils.load_checkpoint(filepath=checkpoint_file)
            self.modelG.load_state_dict(state[ConstVar.KEY_STATE_MODEL_G])
            self.modelD.load_state_dict(state[ConstVar.KEY_STATE_MODEL_D])
            self.optimizerG.load_state_dict(state[ConstVar.KEY_STATE_OPTIMIZER_G])
            self.optimizerD.load_state_dict(state[ConstVar.KEY_STATE_OPTIMIZER_D])
            start_epoch_num = state[ConstVar.KEY_STATE_EPOCH] + 1

        # num epoch 만큼 학습 반복
        for current_epoch_num, count in enumerate(tqdm(range(num_epoch), desc='training process'), start=start_epoch_num):

            # 학습 진행
            self._train()

            # 학습 진행 기록 주기마다 학습 진행 저장 및 시각화
            if (count + 1) % tracking_frequency == 0:

                # 현재 모델을 테스트하기 위한 테스트 객체 생성
                tester = Tester(modelG=deepcopy(x=self.modelG),
                                modelD=deepcopy(x=self.modelD),
                                metric_fn=metric_fn,
                                test_dataloader=test_dataloader,
                                device=self.device)
                tester.running(sample_z_collection=sample_z_collection)

                # 체크포인트 저장
                checkpoint_dir = UtilLib.getNewPath(path=output_dir,
                                                    add=ConstVar.OUTPUT_DIR_SUFFIX_CHECKPOINT)
                checkpoint_filepath = UtilLib.getNewPath(path=checkpoint_dir,
                                                         add=ConstVar.CHECKPOINT_FILE_NAME.format(current_epoch_num))
                utils.save_checkpoint(filepath=checkpoint_filepath,
                                      modelG=self.modelG,
                                      modelD=self.modelD,
                                      optimizerG=self.optimizerG,
                                      optimizerD=self.optimizerD,
                                      epoch=current_epoch_num,
                                      is_best=False)

                # 그래프 시각화 진행
                self._draw_graph(score=tester.score,
                                 current_epoch_num=current_epoch_num,
                                 title=metric_fn.__name__)

                # 결과물 시각화 진행
                pics_dir = UtilLib.getNewPath(path=output_dir,
                                              add=ConstVar.OUTPUT_DIR_SUFFIX_PICS)
                pics_filepath = UtilLib.getNewPath(path=pics_dir,
                                                   add=ConstVar.PICS_FILE_NAME.format(current_epoch_num))
                utils.save_pics(pics_list=tester.pics_list,
                                filepath=pics_filepath,
                                title=self.modelG.__class__.__name__)

    def _train(self):
        """
        * 학습 진행
        :return: 1 epoch 만큼 학습 진행
        """

        # 각 모델을 학습 모드로 전환
        self.modelG.train()
        self.modelD.train()

        # x shape: (N, 3, 64, 64)
        # y shape: (N)
        for x in tqdm(self.train_dataloader, desc='train dataloader', leave=False):

            # 현재 배치 사이즈
            batch_size = x.shape[0]

            # real image label
            real_label = torch.ones(batch_size, device=self.device)
            # fake image label
            fake_label = torch.zeros(batch_size, device=self.device)

            # noise z
            z = torch.randn(size=(batch_size, 100, 1, 1), device=self.device)

            # 텐서를 해당 디바이스로 이동
            x = x.to(self.device)

            # 판별자 학습
            self.modelD.zero_grad()
            # real image 로 학습
            output = self.modelD(x)
            lossD_real = self.loss_fn(output, real_label)
            lossD_real.backward()
            # fake image 로 학습
            fake_x = self.modelG(z)
            output = self.modelD(fake_x.detach())
            lossD_fake = self.loss_fn(output, fake_label)
            lossD_fake.backward()
            self.optimizerD.step()

            # 생성자 학습
            self.modelG.zero_grad()
            fake_x = self.modelG(z)
            output = self.modelD(fake_x)
            lossG = self.loss_fn(output, real_label)
            lossG.backward()
            self.optimizerG.step()

    def _check_is_best(self, tester, best_checkpoint_dir):
        """
        * 현재 저장하려는 모델이 가장 좋은 성능의 모델인지 여부 확인
        :param tester: 현재 모델의 성능을 테스트하기 위한 테스트 객체
        :param best_checkpoint_dir: 비교할 best 체크포인트 파일 디렉터리 위치
        :return: True / False
        """

        # best 성능 측정을 위해 초기화
        try:
            self.best_score
        except AttributeError:
            checkpoint_file = UtilLib.getNewPath(path=best_checkpoint_dir,
                                                 add=ConstVar.CHECKPOINT_BEST_FILE_NAME)
            # 기존에 측정한 best 체크포인트가 있으면 해당 score 로 초기화
            if UtilLib.isExist(checkpoint_file):
                best_tester = deepcopy(x=tester)
                best_tester.running(checkpoint_file=checkpoint_file)
                self.best_score = best_tester.score
            # 없다면 임의의 큰 숫자 (100000) 로 초기화
            else:
                self.best_score = ConstVar.INITIAL_BEST_BCE_LOSS

        # best 성능 갱신
        if tester.score < self.best_score:
            self.best_score = tester.score
            return True
        else:
            return False

    def _draw_graph(self, score, current_epoch_num, title):
        """
        * 학습 진행 상태 실시간으로 시각화
        :param score: 성능 평가 점수
        :param current_epoch_num: 현재 에폭 수
        :param title: 그래프 제목
        :return: visdom 으로 시각화 진행
        """

        # 서버 켜기
        try:
            self.vis
        except AttributeError:
            self.vis = visdom.Visdom()
        # 실시간으로 학습 진행 상태 그리기
        try:
            self.vis.line(Y=torch.cat((torch.Tensor([score[ConstVar.KEY_SCORE_G]]).view(-1, 1), torch.Tensor([score[ConstVar.KEY_SCORE_D]]).view(-1, 1)),
                                      dim=1),
                          X=torch.cat((torch.Tensor([current_epoch_num]).view(-1, 1), torch.Tensor([current_epoch_num]).view(-1, 1)),
                                      dim=1),
                          win=self.plt,
                          update='append',
                          opts=dict(title=title,
                                    legend=['G loss','D loss'],
                                    showlegend=True))
        except AttributeError:
            self.plt = self.vis.line(Y=torch.cat((torch.Tensor([score[ConstVar.KEY_SCORE_G]]).view(-1, 1), torch.Tensor([score[ConstVar.KEY_SCORE_D]]).view(-1, 1)),
                                                 dim=1),
                                     X=torch.cat((torch.Tensor([current_epoch_num]).view(-1, 1), torch.Tensor([current_epoch_num]).view(-1, 1)),
                                                 dim=1),
                                     opts=dict(title=title,
                                               legend=['G loss','D loss'],
                                               showlegend=True))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
