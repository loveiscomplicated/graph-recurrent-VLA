"""Check local HF authentication settings without printing tokens or using the network.

Run from the repository root: python -m scripts.check_huggingface_auth
"""
import os
import sys

from utils.experiment_artifacts import load_hf_env


def main():
    print('기존 HF_TOKEN 환경변수:', bool(os.environ.get('HF_TOKEN', '').strip()))
    try:
        print('HF 인증 파일 로드:', load_hf_env())
        from huggingface_hub import get_token
        configured = bool(get_token())
    except ImportError:
        print('확인 실패: requirements-tracking.txt의 의존성을 설치하세요.', file=sys.stderr)
        return 1
    except Exception as exc:
        # Exceptions can contain credentials; only report the type.
        print(f'확인 실패: 인증 설정을 읽을 수 없습니다 ({type(exc).__name__}).', file=sys.stderr)
        return 1

    print('사용 가능한 HF token:', configured)
    if not configured:
        print('HF_TOKEN, /root/.secrets/hf.env 또는 HF_ENV_FILE, hf auth login 설정을 확인하세요.',
              file=sys.stderr)
        return 1
    print('설정 확인 완료. 실제 인증·Organization 쓰기 권한은 --init-repos 결과로 확인하세요.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
