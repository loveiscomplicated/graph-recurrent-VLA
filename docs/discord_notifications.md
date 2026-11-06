# tmux 로그를 Discord로 전달하기

독립 notifier가 지정한 tmux 세션의 pane 출력과 선택한 로그 디렉터리를 감시한다.
훈련·평가와 같은 호스트에서 실행하며 Python 3.11과 tmux만 필요하다.
공용 `ExperimentLogger`의 로컬/W&B 기록과 병행해서 쓰는 운영 도구다.
실험 ID를 만들거나 checkpoint를 게시하지 않는다.

## 실행

Discord 채널의 Incoming Webhook URL을 notifier 프로세스의
`DISCORD_WEBHOOK_URL` 환경 변수로 설정한다. 플랫폼의 비밀 환경변수나 Git 밖의
비밀 파일에서 주입하고 실제 URL을 command line, shell history, config에 적지 않는다.
notifier는 dotenv나 W&B 인증 파일을 자동으로 읽지 않는다.
이미 실행 중인 tmux 서버는 나중에 설정한 shell 환경변수를 자동으로 받지 않을 수
있으므로, **notifier를 실행할 창 안에서** 환경변수가 제공되는지 확인한다.

훈련 세션이 `recurrent`라면 별도의 tmux 창을 열고 아래를 실행한다.
저장소 루트에서 실행하며 notifier 자신의 pane은 자동으로 제외한다.

```bash
tmux new-window -t recurrent -n discord
.venv/bin/python -m tools.discord_notifier \
  --session recurrent \
  --log-dir outputs/recurrent/exp/main/logs
```

`run_all.sh smoke`에는 `outputs/recurrent/exp/smoke/logs`, official 실험에는
실제 config의 `out_dir` 아래 `logs`를 지정한다. Recurrent launcher는 자식
train/eval 출력을 파일로 보내므로, pane 감시와 이 디렉터리 감시를 함께 사용한다.
Python 출력이 파일에 버퍼링되는 실행은 `PYTHONUNBUFFERED=1`을 설정한 환경에서
launcher를 시작하면 더 빨리 전달된다. 이미 실행 중인 프로세스의 버퍼링은 바꾸지 않는다.

Graph v4 matrix 실행의 예:

```bash
.venv/bin/python -m tools.discord_notifier \
  --session graph-v4-mps \
  --log-dir outputs/graph/v4/training \
  --log-dir outputs/graph/v4/evaluation
```

`--log-dir`는 반복 지정할 수 있다. 해당 디렉터리 아래의 `*.log`를 재귀적으로
읽으며 디렉터리가 나중에 생겨도 감시한다. 알림을 원하는 실험의 디렉터리만 지정한다.
pane 출력만 필요하면 `--log-dir`를 생략한다. tmux에 보이는 출력 전체를 감시하므로
선택한 세션은 실험용으로 사용한다. 같은 세션에는 notifier 하나를 실행한다.

## 전송과 실패 처리

- 기본 전송 간격은 60초이며, 여러 소스를 한 메시지로 묶는다. 변경하려면
  `--interval 120`처럼 초를 지정한다. 출력이 없으면 메시지를 보내지 않는다.
- 메시지는 최대 1,800 UTF-16 단위로 제한한다. 로그가 많으면 최근 줄과 생략한 줄 수를
  보낸다. 실패 중에도 대기 메모리는 최근 1,000줄로 제한한다.
- pane과 `tee` 파일에서 2초 이내에 같은 줄이 관측되면 중복 전송을 줄인다.
  같은 소스의 반복 줄은 유지한다. 별도 작업의 우연히 동일한 줄도 합쳐질 수 있다.
- 429 응답은 `Retry-After`, JSON의 `retry_after`와 reset 응답을 따른다.
  네트워크/5xx 오류는 5초부터 최대 300초까지 간격을 늘려 재시도한다.
  그동안 새 로그를 계속 받으며 복구 시 최근 내용을 보낸다.
- 다른 HTTP 오류(잘못된 요청, 권한 문제, 삭제된 웹훅 등)는 전송을 중단한다.
  로컬 수집은 계속하며 웹훅을 수정한 뒤 notifier를 다시 실행한다.
- `Ctrl-C`/SIGTERM 또는 세션 종료 시 수집을 마치고, 전송 제한에 걸리지 않는 경우에만
  마지막 묶음을 시도한다. 진행 중인 HTTP 요청은 기본 5초 timeout을 사용한다.
  네트워크 실패로 마지막 메시지가 누락될 수 있으며, 재시작 시 과거 로그를 재전송하지 않는다.

Discord 제한은 고정된 서버 할당량을 가정하지 않고 응답을 읽어 처리한다.
전송 timeout 뒤 재시도에서는 이미 도착한 메시지가 중복될 수 있다.
공식 규격: [rate limits](https://docs.discord.com/developers/topics/rate-limits),
[execute webhook](https://docs.discord.com/developers/resources/webhook#execute-webhook).

## 로컬 캡처와 tmux 연결

기존 로그 파일은 시작 시 EOF부터 읽고, 새 파일은 처음부터 읽는다.
파일 truncate·교체를 감지하며 원본 파일을 수정하지 않는다. 심볼릭 링크 파일은 건너뛴다.
ANSI 제어 문자는 제거하고 `\r` 진행률 출력은 최신 내용으로 합친다.
개행 없는 출력은 약 2초간 추가 출력이 없을 때 내보낸다. 65,536자를 넘는 한 줄은
내용 대신 생략 표시를 남긴다.

pane 출력은 `outputs/discord_notifier/<실행시각-ID>/panes.log`에도 기록한다.
`--output-dir`로 부모 디렉터리를 변경할 수 있다. 파일 소스는 원본 `.log`가 로컬 기록이다.
캡처 파일은 Git에서 제외되며 자동 삭제·순환하지 않으므로 필요 없는 이전 캡처는
사용자가 정리한다. 시작/종료·일시적인 무출력만으로 훈련 성공이나 실패를 추정하지 않는다.

세션 내 기존·신규 pane에 output-only `pipe-pane`을 연결한다. 기존 pipe가 있는 pane은
안내 후 건너뛴다. 정상 종료 시 연결 중인 자신의 pipe를 해제한다. notifier가 강제 종료돼도
연결용 보조 프로세스는 종료된다. 이때 tmux의 닫힌 pipe 표시는 다음 pane 출력 때까지
남을 수 있으며 새 notifier는 표시가 사라진 뒤 다시 연결한다. 다른 pipe, 세션,
훈련 프로세스에는 종료 명령을 보내지 않는다.
로컬 캡처 디스크 쓰기 실패는 notifier를 종료시키며 훈련 프로세스는 계속 실행된다.

웹훅 URL과 현재 환경변수 중 API key/token/password/secret/credential/webhook 이름을
가진 값은 전송·캡처 전에 가린다. 환경에 없는 임의의 비밀까지 인식하지는 못하므로
훈련 자체가 비밀을 출력하지 않게 유지한다. Discord 멘션은 비활성화하며, HTTP 오류에는
응답 본문이나 URL을 출력하지 않는다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_discord_notifier -v
```

HTTP 응답을 모의 처리하므로 실제 Discord에 쓰지 않는다. tmux가 있으면 별도 이름의
임시 서버에서 수집·신규 pane·기존 pipe 보존·종료 정리를 검증한 뒤 서버를 삭제한다.
