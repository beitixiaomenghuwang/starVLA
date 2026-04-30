# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


from abc import ABC, abstractmethod

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
# from gr00t.model.transforms import GR00TTransform


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


###########################################################################################

class OxeDroidDataConfig:
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_position": "binary",
                },
                target_rotations={"action.eef_rotation_delta": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class OxeBridgeDataConfig:
    video_keys = [
        "video.image_0",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.roll": "q99",
                    "state.pitch": "q99",
                    "state.yaw": "q99",
                    "state.pad": "q99",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "binary",
                },
            ),
            # concat transforms
            # ConcatTransform(
            #     # video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class OxeRT1DataConfig:
    video_keys = [
        "video.image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.rx",
        "state.ry",
        "state.rz",
        "state.rw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.rx": "q99",
                    "state.ry": "q99",
                    "state.rz": "q99",
                    "state.rw": "q99",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "binary",
                },
            ),
            # concat transforms
            # ConcatTransform(
            #     # video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class SingleFrankaRobotiqDeltaEefDataConfig:
    video_keys = [
        "video.base_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
    ]
    action_keys = [
        "action.delta_eef_position",
        "action.delta_eef_rotation",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.eef_rotation": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_eef_position": "min_max",
                    "action.delta_eef_rotation": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class Libero4in1DataConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
            apply_to=self.action_keys,
            normalization_modes={
                "action.x": "min_max",
                "action.y": "min_max",
                "action.z": "min_max",
                "action.roll": "min_max",
                "action.pitch": "min_max",
                "action.yaw": "min_max",
            },
        ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################


class SingleFrankaRobotiqDeltaJointsDataConfig:
    video_keys = [
        "video.base_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.joints",
    ]
    action_keys = [
        "action.delta_joints",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joints": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_joints": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class FourierGr1ArmsWaistDataConfig:
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


###########################################################################################

class SO101Config:
    #input
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    
    state_keys = [
        "state.shoulder_pan.pos",
        "state.shoulder_lift.pos",
        "state.elbow_flex.pos",
        "state.wrist_flex.pos",
        "state.wrist_roll.pos",
        "state.gripper.pos",
    ]
    language_keys = ["annotation.human.action.task_description"]

    # output
    action_keys = [
        "action.shoulder_pan.pos",
        "action.shoulder_lift.pos",
        "action.elbow_flex.pos",
        "action.wrist_flex.pos",
        "action.wrist_roll.pos",
        "action.gripper.pos",
    ]
    

    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    key: "min_max" for key in self.state_keys
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    key: "min_max" for key in self.action_keys
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)



class ArxX5DataConfig:
    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",
        "action.left_gripper",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max",
                    "state.right_joints": "min_max",
                    "state.left_gripper": "binary",
                    "state.right_gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max",
                    "action.right_joints": "min_max",
                    "action.left_gripper": "binary",
                    "action.right_gripper": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################


class AgilexDataConfig:
    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",#@JinhuiYE this order is different from Dataset
        "action.left_gripper",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max",
                    "state.right_joints": "min_max",
                    "state.left_gripper": "binary",
                    "state.right_gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max",
                    "action.right_joints": "min_max",
                    "action.left_gripper": "binary",
                    "action.right_gripper": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################


class AgilexData50Config:
    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",  # @JinhuiYE this order is different from Dataset
        "action.left_gripper",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "state.left_joints": "min_max",
                    "state.right_joints": "min_max",
                    "state.left_gripper": "binary",
                    "state.right_gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "action.left_joints": "min_max",
                    "action.right_joints": "min_max",
                    "action.left_gripper": "binary",
                    "action.right_gripper": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class TeleAvatarDataConfig(BaseDataConfig):
    """
    Data config for the TeleAvatar dual-arm robot (teleavatar).

    Raw sub-keys loaded from parquet (before Rotate6D conversion):
      left_ee_pose  (7): pos_xyz(3) + quat_xyzw(4)
      right_ee_pose (7): pos_xyz(3) + quat_xyzw(4)
      left_gripper_effort  (1)
      right_gripper_effort (1)

    After TeleAvatarDataset.__getitem__ applies Rotate6D + q99 normalisation:
      Action  (20-dim): [pos_L(3) | rot6d_L(6) | grip_L(1) | pos_R(3) | rot6d_R(6) | grip_R(1)]
      State   (18-dim): [pos_L(3) | rot6d_L(6) | pos_R(3)  | rot6d_R(6)]

    Normalisation:
      EE position:     q99  → [-1, 1]  (using dataset statistics)
      Rot6D:           none (bounded in [-1, 1] by construction)
      Gripper effort:  q99  → {≈-1, ≈+1}  (two-valued → effectively binary)
        left:  q01≈-0.5, q99≈2.5  ⟹  close→-1,  open→+1
        right: q01≈-2.5, q99≈0.5  ⟹  open→-1,  close→+1
      NOTE: do NOT use 'binary' (threshold=0.5): right close=0.5 would map
      to 0, same as right open=-2.5, making the two states indistinguishable.

    The lerobot_modality_meta is defined programmatically via get_lerobot_modality_meta()
    and written to meta/modality.json at dataset load time — no hand-crafted JSON needed.

    Index mapping in the raw 62-dim action / observation.state arrays:
      [0:8]   left  joint positions (7) + left  gripper position (1)
      [8:16]  right joint positions (7) + right gripper position (1)
      [16:24] left  joint velocities       [24:32] right joint velocities
      [32:40] left  joint efforts          [40:48] right joint efforts
        ↳ left_gripper_effort  @ index 39
        ↳ right_gripper_effort @ index 47
      [48:55] left  EE position_xyz(3) + orientation_xyzw(4)
      [55:62] right EE position_xyz(3) + orientation_xyzw(4)
    """

    video_keys = [
        "video.left_color",
        "video.right_color",
        "video.chest_camera",
    ]
    state_keys = [
        "state.left_ee_pose",
        "state.right_ee_pose",
    ]
    action_keys = [
        "action.left_ee_pose",
        "action.left_gripper_effort",
        "action.right_ee_pose",
        "action.right_gripper_effort",
    ]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))    # predict 16 future steps
    state_indices = [0]                 # current end-effector state

    @staticmethod
    def get_lerobot_modality_meta() -> dict:
        """
        Return the modality metadata dict that is written to meta/modality.json.

        Keeping this in Python code means the JSON is a generated artifact —
        edit this function instead of the JSON file.
        """
        return {
            "action": {
                # left_ee_pose: left_ee_position_x/y/z + left_ee_orientation_x/y/z/w (7 dims)
                "left_ee_pose": {
                    "start": 48,
                    "end": 55,
                    "original_key": "action",
                },
                # left_gripper_effort: 1 dim (index 39 in raw 62-dim array)
                "left_gripper_effort": {
                    "start": 39,
                    "end": 40,
                    "original_key": "action",
                },
                # right_ee_pose: right_ee_position_x/y/z + right_ee_orientation_x/y/z/w (7 dims)
                "right_ee_pose": {
                    "start": 55,
                    "end": 62,
                    "original_key": "action",
                },
                # right_gripper_effort: 1 dim (index 47 in raw 62-dim array)
                "right_gripper_effort": {
                    "start": 47,
                    "end": 48,
                    "original_key": "action",
                },
            },
            "state": {
                "left_ee_pose": {
                    "start": 48,
                    "end": 55,
                    "original_key": "observation.state",
                },
                "right_ee_pose": {
                    "start": 55,
                    "end": 62,
                    "original_key": "observation.state",
                },
            },
            "video": {
                "left_color":   {"original_key": "observation.images.left_color"},
                "right_color":  {"original_key": "observation.images.right_color"},
                "chest_camera": {"original_key": "observation.images.chest_camera"},
            },
            "annotation": {
                "human.action.task_description": {"original_key": "task_index"},
            },
        }

    def modality_config(self) -> dict:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        # NOTE: the current __getitem__ fast-path bypasses ComposedModalityTransform,
        # so these transforms are not applied during training.
        # They are kept here for documentation and for future re-enabling.
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # action transforms (NOT currently applied — see module-level note above)
            # Actual normalization is handled by TeleAvatarDataset.__getitem__:
            #   EE pose      → q99 for position, no-op for rot6d (already bounded)
            #   Gripper effort → q99 (NOT binary: right close=0.5 ≤ threshold 0.5 → indistinguishable)
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_ee_pose":         "q99",
                    "action.left_gripper_effort":  "q99",
                    "action.right_ee_pose":        "q99",
                    "action.right_gripper_effort": "q99",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
    
    
ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "oxe_droid": OxeDroidDataConfig(),
    "oxe_bridge": OxeBridgeDataConfig(),
    "oxe_rt1": OxeRT1DataConfig(),
    "SO101": SO101Config(),
    "demo_sim_franka_delta_joints": SingleFrankaRobotiqDeltaJointsDataConfig(),
    "arx_x5": ArxX5DataConfig(),
    "robotwin": AgilexDataConfig(),
    "robotwin50": AgilexData50Config(),
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "vla_arena_franka": VLAArenaFrankaDataConfig(),

    "custom_robot_config": SingleFrankaRobotiqDeltaEefDataConfig(),
    "teleavatar": TeleAvatarDataConfig(),
}
