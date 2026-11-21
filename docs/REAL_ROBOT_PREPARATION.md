# Real G1 preparation — execution is not enabled

The prototype only starts MuJoCo. There is no real-robot launch command or
automatic switch from simulator to hardware.

Before a local operator integrates the upstream real-world transport, obtain
the exact G1 joint/hand configuration, actuator/firmware limits, camera model,
camera-to-body calibration, clock synchronization and root-localization source.
The current rubber-hand simulation cannot establish Dex3 finger capabilities.

Use the versioned program/knowledge snapshot selected in simulation, the pinned
ScaleBFM checkpoint, the verified metadata order and one fixed supported control
mode. The operator must determine whether an eager policy is suitable for the
deployment runtime or compile/test TensorRT on the target platform. The current
simulation runtime is not a deployment latency guarantee.

Prepare a camera adapter that produces the same timestamped RGB/optional-depth
contract with intrinsics, optical-frame extrinsics and proprioception. Missing
depth or uncertain pose must remain unknown. Replace ideal simulation root
localization with a verified hardware source; do not silently treat simulator
calibration as physical calibration.

The first operator-controlled commissioning activities are stationary tracking,
small free-space wrist motion and body-height changes, with measured reference
error and the platform's normal supervision. Contact tasks depend on successful
simulation validation and the site's hardware commissioning procedures. No
successful real-robot transfer is claimed in this delivery.
