"""Check W&B upload from the current Workspace without printing credentials.

Run from the repository root: python -m scripts.check_experiment_tracking
"""
import sys

from utils.experiment_logger import ExperimentLogger


def main():
    try:
        with ExperimentLogger(
            team='integration',
            job_type='smoke',
            config={'purpose': 'vessl_tracking_check'},
        ) as logger:
            for key in ('wandb_mode', 'entity', 'project', 'group', 'run_id'):
                print(f'{key}: {logger.metadata[key]}', flush=True)
            print(f'로컬 기록: {logger.directory}', flush=True)

            if logger.metadata['wandb_mode'] != 'online':
                raise RuntimeError(
                    'Offline 상태입니다. 인증 파일의 WANDB_API_KEY와 '
                    'WANDB_ENTITY, 기존 WANDB_MODE 환경변수를 확인하세요.'
                )
            if logger.run is None:
                raise RuntimeError(
                    'W&B 초기화 실패: SDK 설치, 네트워크, API key와 '
                    '프로젝트 기록 권한을 확인하세요.'
                )
            logger.log({'tracking/connection_check': 1})
            if logger.metadata.get('tracking_errors'):
                raise RuntimeError('W&B metric 기록에 실패했습니다.')
    except RuntimeError as exc:
        print(f'확인 실패: {exc}', file=sys.stderr)
        return 1

    if logger.metadata.get('tracking_errors'):
        print('확인 실패: W&B run 종료 중 오류가 발생했습니다.', file=sys.stderr)
        return 1

    print(f'W&B 전송 완료: {logger.metadata.get("wandb_url")}')
    print('대시보드에서 tracking/connection_check = 1을 확인하세요.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
