# PegInsertionSide — Subtask Specification v0

## 1. Reach

**목표:** 로봇의 end-effector(EE)를 peg를 잡기 위한 위치와 방향으로 이동시키는 task입니다.

- **시작 조건:** Peg가 테이블 위에 있고, 로봇의 gripper가 열린 상태. 기본 ManiSkill 초기 상태에서 시작합니다.
- **목표 상태:** EE가 peg의 grasp pose 앞에 있는 접근 위치(reach pose)에 도달합니다.
- **성공 조건:** EE와 목표 reach pose 사이의 위치·방향 오차가 허용 범위 이내이고, peg는 아직 집지 않은 상태입니다.
- **실패 조건:** 제한 시간 초과, peg가 크게 밀려나거나 도달할 수 없는 상태가 된 경우.

초기 성공 판정 후보는 위치 오차 2cm 이내, 회전 오차 15° 이내입니다.

Gripper는 열린 상태로 고정해도 됩니다. 접촉이나 물체 이동이 거의 필요하지 않기 때문에 **첫 번째 BC baseline을 확보하기에 가장 적절한 task**로 생각합니다.

## 2. Grasp

**목표:** Peg에 접근해 gripper로 안정적으로 잡는 task입니다.

- **시작 조건:** Reach가 완료되어 EE가 peg 근처에 있고, gripper는 열려 있는 상태입니다.
- **목표 상태:** Gripper가 peg의 적절한 위치를 잡고 있는 상태입니다.
- **성공 조건:** ManiSkill의 `is_grasping` 판정이 참이고, 짧은 시간 동안 peg와 gripper 사이의 상대 위치가 안정적으로 유지됩니다.
- **실패 조건:** 제한 시간 초과, grasp 실패, peg를 밀어내거나 놓치는 경우.

이번에는 grasp 동작 자체에 집중합니다. Peg를 hole까지 이동시키는 과정은 포함하지 않습니다.

다만 `is_grasping`만으로 실제로 안정적인 grasp가 이루어졌는지 완전히 확인하기 어려울 수 있으므로, 필요하면 짧은 이동 중 peg가 유지되는지도 추가로 검사합니다.

## 3. Transport

**목표:** 이미 잡은 peg를 hole 근처의 pre-insertion 위치로 이동시키는 task입니다.

- **시작 조건:** Peg를 안정적으로 grasp한 상태이고, 아직 hole에서 떨어져 있습니다.
- **목표 상태:** Peg가 hole 입구 앞의 대략적인 목표 위치와 방향에 도달합니다.
- **성공 조건:** Peg가 pre-insertion 목표 pose 근처에 도달하고 grasp 상태가 유지됩니다. 정밀한 삽입 정렬까지는 요구하지 않습니다.
- **실패 조건:** 제한 시간 초과, 이동 중 peg를 놓침, workspace 이탈.

초기에는 목표 pose와의 위치 오차 3~5cm, 방향 오차 20° 정도를 허용하는 후보를 생각할 수 있습니다.

이 task에서는 peg와 hole의 상대 위치·방향이 중요해집니다. 따라서 Reach보다 Graph representation의 효과를 확인하기에 적합할 것으로 예상합니다.

초기 실험에서는 gripper를 닫힌 상태로 유지해 이동 제어만 검증할 수도 있습니다.

## 4. Align

**목표:** Hole 근처로 이동한 peg를 실제로 삽입할 수 있도록 정밀하게 정렬하는 task입니다.

- **시작 조건:** Transport가 완료되어 peg가 hole 입구 근처에 있고, grasp 상태가 유지되고 있습니다.
- **목표 상태:** Peg의 삽입축이 hole의 삽입축과 정렬되고, peg head가 hole 입구 앞의 적절한 위치에 놓입니다.
- **성공 조건:** Hole-local 좌표계에서 peg head와 peg 중심의 횡방향 오차, 상대 회전 오차, 삽입축 방향의 대기 위치가 모두 허용 범위 이내입니다.
- **실패 조건:** 제한 시간 초과, peg를 놓침, 심한 충돌 또는 정렬 불가능한 상태.

여기서는 물체 중심 사이의 거리만으로 성공을 판단하면 안 됩니다.

Peg의 방향이 틀어져 있으면 중심이 가까워도 삽입에 실패할 수 있기 때문입니다.

기존 ManiSkill reward에는 peg head와 중심의 횡방향 오차가 각각 1cm 미만인지를 확인하는 조건이 있습니다. 다만 실제 hole clearance는 3mm이므로, **연구용 Align 성공 조건은 상대 회전과 더 정밀한 표면 정렬까지 고려**하는 방향으로 정의합니다.

정확한 허용 오차는 expert trajectory를 분석한 뒤 확정합니다.

## 5. Insert

**목표:** 정렬된 peg를 hole 안쪽으로 밀어 넣는 task입니다.

- **시작 조건:** Align이 완료되어 peg가 hole 입구 앞에 정렬되어 있고, grasp 상태가 유지되고 있습니다.
- **목표 상태:** Peg가 hole 내부로 충분히 삽입된 상태입니다.
- **성공 조건:** 우선 ManiSkill 공식 `evaluate()`의 insertion success 판정을 사용합니다.
- **실패 조건:** 제한 시간 초과, 삽입 중 심한 충돌로 진행 불가, peg를 놓침.

이 task는 정렬이 완료된 상태에서 시작하므로, 주로 삽입축을 따라 움직이면서 미세한 오차를 제어하는 문제에 집중합니다.

공식 성공 판정 외에 insertion depth, lateral error, collision 여부도 별도로 기록하면 좋겠습니다.

---

## 공통 설계 원칙

**1. 모든 Subtask는 독립적으로 실행할 수 있어야 합니다.**

Reach는 기본 초기 상태에서 시작하지만, 나머지는 기존 expert trajectory의 특정 시점부터 시작할 수 있어야 합니다. 초기에는 검증된 expert prefix replay를 이용하고, 이후 다양한 초기 위치와 방향을 sampling하는 방식으로 확장합니다.

**2. 성공 조건은 실제 simulator state로 판단합니다.**

각 단계의 성공 여부를 사람이 직접 확인하는 것이 아니라, EE pose, peg/hole pose, grasp state 등을 이용해 자동으로 계산합니다.

수치 threshold는 위의 초기 후보를 그대로 확정하지 않고, 기존 expert demonstration에서 해당 조건이 실제로 달성되는지 먼저 확인한 후 고정합니다.

**3. Subtask의 성공과 다음 Subtask의 시작은 연결되어야 합니다.**

예를 들어 Transport의 성공 상태는 Align의 유효한 시작 상태여야 합니다. 다만 각각을 독립적으로 학습하고 평가할 수 있도록 만듭니다.

**4. 원본 Trajectory는 그대로 보존합니다.**

각 subtask 데이터에는 최소한 다음을 저장합니다.

```text
subtask_id
episode_id
initial_state
states             # T+1
actions            # T
goal_metadata
success / failure
termination_reason
```

Robot/object의 GT geometry와 pose도 재구성 가능하게 보존합니다. 나중에 BC 외의 학습 방식으로 확장할 때도 같은 데이터를 재사용하기 위해서입니다.

**5. 이번에는 Behavior Cloning만 수행합니다.**

각 subtask에 대해 expert data를 생성하고, 기존 Flat MLP baseline을 학습해 실제 rollout SR을 측정합니다. 그다음 같은 조건에서 GNN을 추가해 성능 차이를 비교합니다.

---

## 구현 및 검증 순서

우선 다섯 개 subtask의 조건을 코드로 정의하고, 기존 91개 expert episode에서 각 단계의 시작·성공 조건이 실제로 관측되는지 확인합니다.

이후 각 subtask를 독립적으로 reset하고 데이터를 생성할 수 있는 pipeline을 구현합니다.

첫 BC 실험은 **Reach**에서 진행합니다. Reach에서 baseline이 안정적으로 작동하면, Transport와 Align 등 관계 표현이 더 중요한 task로 확장합니다.

지금 단계에서는 새로운 Goal Graph encoder나 GCSL 학습을 구현하지 않습니다.