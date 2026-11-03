from common import *
import traceback
v=rd(R/'verification.json') if (R/'verification.json').exists() else {}
v.update(stage='RUNTIME_STOPPED',last_stage=v.get('stage'),automatic_retry=False)
wr('verification.json',v)
print('RUNTIME_STOPPED: results preserved; no automatic retry')
