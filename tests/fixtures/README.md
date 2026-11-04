# Current-live baseline fixture

`live_policy_d222fe2.json` was generated before modularization at source commit
`d222fe2bd555b4c6c1bc6e6fb65c5017bbe242d5`, with Torch2.14.1 on this macOS CPU host.
It stores synthetic Graph21/robot76 inputs, model seed42, E0/E1 state_dict names
and SHA256, action/state outputs and gradient norms. It contains no dataset,
checkpoint weights, credentials or API response.

`tests.test_modular_live` checks exact CPU equality on the verified host. Floating
point kernels/Torch versions can differ on other hosts; such differences require
comparison with the same historical implementation there, rather than updating
the fixture blindly. The fixture does not establish CUDA equivalence or speed.
