import numpy as np
import cv2
import glog as log
class Reward():
    '''
    define different type reward function
    '''
    def __init__(self, setting):
        self.reward_factor = setting['reward_factor']
        self.reward_th = setting['reward_th']
        self.dis2target_last = 0
        self.stepscale = setting['stepscale']

    # IN: object mask delivered by unrealcv client, mask, and pose
    # OUT: reward

    def reward_mask(self, pose, slope, roughness,
                    factor,
                    right_shift_one,
                    right_shift_two,
                    stretch_one,
                    stretch_two):
        reward = 0

        mask_score = 0

        #height, width = mask.shape
        #tot_num_pixels = height*width
        #fov_score = (cv2.sumElems(mask)[0] / 255) / tot_num_pixels
        log.warn("Mask Score: {}".format(mask_score))

        reward = factor*(np.tanh((1/stretch_one)*(2*np.pi*mask_score-right_shift_one*np.pi)) +
                         np.tanh((1/stretch_two)*(2*np.pi*mask_score-right_shift_two*np.pi)))
        log.warn("Reward for Mask: {}".format(reward))

        return reward, mask_score

    def reward_height(self, mesh_height, pose,
                      scale,
                      stretch):
        reward = 0
        height = pose[2]
        distance = height - mesh_height

        #log.warn("Height for reward height {}".format(distance))

        interim = scale*np.tanh((1/stretch)*distance)
        #log.warn("Interim Height: {}".format(interim))

        reward  = (-1)*np.max(np.asarray([0,interim]))  # delete max to consider negative values
        log.warn("Reward Height: {}".format(reward))

        return reward, distance

    def reward_distance(self, dis2target_now):
        reward = 10*(self.dis2target_last - dis2target_now) / max(self.dis2target_last/10, 200)
        self.dis2target_last = dis2target_now
        log.warn("Reward Distance: {}".format(reward))
        return reward

    def reward_mask_height(self, pose,  error, mesh_height, slope, roughness, landable, dist_landable, scale, stretch, done_thr, success_thr,
                           factor=100,
                           right_shift_one=1,
                           right_shift_two=1.5,
                           stretch_one=9,
                           stretch_two=2): #scale 300, stretch 3000
        done    = False
        success = False
        reward  = 0
        #reward_fov, mask_score = self.reward_mask(pose, slope, roughness, factor, right_shift_one, right_shift_two, stretch_one, stretch_two)
        reward_height, distance = self.reward_height(mesh_height, pose, scale, stretch)
        reward_distance = self.reward_distance(distance)

        reward = reward_distance + reward_height #+ reward_FOV

        # Adding a step reward readded
        if error > 5 * self.stepscale:
            done = True
            reward -= 10000
            log.warn("OUT OF BOUNDARIES")
        else:
            if distance < done_thr:
                done = True
                
                #here slope, roughness reward
                if landable == 1:
                    reward += 500
                    log.warn("SUCCESS landed correctly")
                    success = True
                else:
                    log.warn("FAILED landed incorrectly")
                    reward -= 500

            if dist_landable < 100: #100 is like threshold that can be modified
            reward += 10

        

        #log.warn("Reward Total: {}".format(reward))
        return reward, done, success

    def reward_sinc(self, mask, pose, done_thr, success_thr,
                        normalization_factor=3000,
                        scale=10):
        reward = 0
        normalized_height = pose[2]/normalization_factor

        height, width = mask.shape
        tot_num_pixels = height*width

        fov_score = (cv2.sumElems(mask)[0] / 255) / tot_num_pixels
        log.warn("FOV Score: {}".format(fov_score))

        reward = scale*np.sinc(4*((fov_score-1)**2+ normalized_height**2) / np.pi) - 10

        if pose[2] < done_thr:
            done = True
            if fov_score > success_thr:
                reward += 100
                log.warn("SUCCESS")
                success = True
            else:
                reward -= 100

        log.warn("Reward Sinc Total: {}".format(reward))

        return reward

    def reward_temporal(self, mask, pose, done_thr, success_thr,
                        normalization_factor=3000,
                        scale=10):
        pass

    def reward_depth(self, depth):
        pass

    def reward_bbox(self, boxes):
        reward = 0
        # summarize the reward of each detected box
        for box in boxes:
            reward += self.get_bbox_reward(box)

        if reward > self.reward_th:
            # get ideal target
            reward = min(reward * self.reward_factor, 10)
        elif reward == 0:
            # false trigger
            reward = -1
        else:
            # small target
            reward = 0

        return reward, boxes

    def get_bbox_reward(self, box):
        # get reward of single box considering the size and position of box
        (xmin, ymin), (xmax, ymax) = box
        boxsize = (ymax - ymin) * (xmax - xmin)
        x_c = (xmax + xmin) / 2.0
        x_bias = x_c - 0.5
        discount = max(0, 1 - x_bias ** 2)
        reward = discount * boxsize
        return reward
