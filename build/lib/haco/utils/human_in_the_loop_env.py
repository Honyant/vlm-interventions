import copy

import numpy as np
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.engine.engine_utils import get_global_config
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.manual_control_policy import TakeoverPolicy
from metadrive.utils.math_utils import safe_clip

ScreenMessage.SCALE = 0.1

class MyKeyboardController(KeyboardController):
    # Update Parameters
    STEERING_INCREMENT = 0.05
    STEERING_DECAY = 0.5

    THROTTLE_INCREMENT = 0.5
    THROTTLE_DECAY = 1

    BRAKE_INCREMENT = 0.5
    BRAKE_DECAY = 1

class MyTakeoverPolicy(TakeoverPolicy):
    """
    Record the takeover signal
    """
    def __init__(self, obj, seed):
        super(TakeoverPolicy, self).__init__(obj, seed)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "joystick":
                self.controller = SteeringWheelController()
            elif config["controller"] == "keyboard":
                self.controller = MyKeyboardController(False)
            else:
                raise ValueError("Unknown Policy: {}".format(config["controller"]))
        self.takeover = False


class HumanInTheLoopEnv(SafeMetaDriveEnv):
    """
    This Env depends on the new version of MetaDrive
    """

    def default_config(self):
        config = super(HumanInTheLoopEnv, self).default_config()
        config.update(
            {
                "environment_num": 50,
                "start_seed": 100,
                "cost_to_reward": True,
                "traffic_density": 0.06,
                "manual_control": False,
                "controller": "joystick",
                "agent_policy": MyTakeoverPolicy,
                "only_takeover_start_cost": True,
                "main_exp": True,
                "random_spawn": False,
                "cos_similarity": True,
                "out_of_route_done": True,
                "in_replay": False,
                "random_spawn_lane_index": False,
                "target_vehicle_configs": {
                    "default_agent": {"spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 1)}}
            },
            allow_add_new_key=True
        )
        return config

    def reset(self, *args, **kwargs):
        self.in_stop = False
        self.t_o = False
        self.total_takeover_cost = 0
        self.input_action = None
        ret = super(HumanInTheLoopEnv, self).reset(*args, **kwargs)
        if self.config["random_spawn"]:
            self.config["vehicle_config"]["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2,
                                                                 self.engine.np_random.randint(3))
        # keyboard is not as good as steering wheel, so set a small speed limit
        self.vehicle.update_config({"max_speed": 25 if self.config["controller"] == "keyboard" else 40})
        return ret

    def _get_step_return(self, actions, engine_info):
        o, r, d, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        if self.config["in_replay"]:
            return o, r, d, engine_info
        controller = self.engine.get_policy(self.vehicle.id)
        last_t = self.t_o
        self.t_o = controller.takeover if hasattr(controller, "takeover") else False
        engine_info["takeover_start"] = True if not last_t and self.t_o else False
        engine_info["takeover"] = self.t_o
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.t_o

        if not condition:
            self.total_takeover_cost += 0
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost

        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["total_native_cost"] = self.episode_cost
        return o, r, d, engine_info

    def _is_out_of_road(self, vehicle):
        ret = (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def step(self, actions):
        self.input_action = copy.copy(actions)
        ret = super(HumanInTheLoopEnv, self).step(actions)
        while self.in_stop:
            self.engine.taskMgr.step()
        if self.config["use_render"] and self.config["main_exp"] and not self.config["in_replay"]:
            super(HumanInTheLoopEnv, self).render(text={
                "Steering": actions[0],
                "Acceleration": actions[1],
                "Intervention occuring": self.t_o,
                # "Stop (Press E)": ""
            })
        return ret

    def stop(self):
        self.in_stop = not self.in_stop

    def setup_engine(self):
        super(HumanInTheLoopEnv, self).setup_engine()
        self.engine.accept("e", self.stop)

    def get_takeover_cost(self, info):
        if not self.config["cos_similarity"]:
            return 1
        takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action = safe_clip(np.array(self.input_action), -1, 1)
        # cos_dist = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1]) / 1e-6 +(
        #         np.linalg.norm(takeover_action) * np.linalg.norm(agent_action))

        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident

        return 1 - cos_dist


if __name__ == "__main__":
    env = HumanInTheLoopEnv(
        {"manual_control": True, "disable_model_compression": True, "use_render": True, "main_exp": True})
    env.reset()
    while True:
        env.step([0, 0])
