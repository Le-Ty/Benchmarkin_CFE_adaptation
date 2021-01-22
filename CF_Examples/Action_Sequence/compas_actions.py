import tensorflow as tf
import numpy as np
from CF_Models.act_seq.actions.action import Action
import CF_Models.act_seq.actions.condition_ops as cond


class WaitYears(Action):
    def __init__(self, features):
        super().__init__(name='WaitYears',
                         description='Wait x amount of years',
                         type='Numeric',
                         features=features,
                         target_features=['age'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features['age'].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_age = self.features['age'].get_feature_value(new_instance, use_tensor,
                                                         space='x')
        old_age = self.features['age'].get_feature_value(instance, use_tensor, space='x')
        cost = tf.abs(new_age - old_age) if use_tensor else np.abs(new_age - old_age)
        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_age = self.features['age'].get_feature_value(new_instance, use_tensor,
                                                         space='x')
        old_age = self.features['age'].get_feature_value(instance, use_tensor, space='x')
        return cond.op_and(cond.op_gt(new_age, old_age, use_tensor, scale=100.),
                           cond.op_lt(new_age, 120, use_tensor, scale=100.), use_tensor)


class ChangeChargeDegree(Action):
    def __init__(self, features):
        super().__init__(name='ChangeChargeDegree',
                         description='change chargeing degree',
                         type='Numeric',
                         features=features,
                         target_features=['r_charge_degree'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        return self.features['r_charge_degree'].change_feature_value(instance,
                                                                     self.get_param(p,
                                                                                    use_tensor),
                                                                     use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_edu = self.features['r_charge_degree'].get_feature_value(new_instance, use_tensor,
                                                                     space='x')
        old_edu = self.features['r_charge_degree'].get_feature_value(instance, use_tensor, space='x')
        cost = tf.abs(new_edu - old_edu) if use_tensor else np.abs(new_edu - old_edu)

        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_edu = self.features['r_charge_degree'].get_feature_value(new_instance, use_tensor,
                                                                     space='x')
        old_edu = self.features['r_charge_degree'].get_feature_value(instance, use_tensor,
                                                                     space='x')
        return cond.op_and(cond.op_and(cond.op_gt(new_edu, 0, use_tensor, scale=100.),
                                       cond.op_lt(new_edu, 3.1, use_tensor, scale=100.),
                                       use_tensor),
                           cond.op_and(cond.op_gt(old_edu, new_edu, use_tensor, scale=10.),
                                       cond.op_gt(new_edu, 0, use_tensor, scale=10.),
                                       use_tensor),
                           use_tensor)


class ChangeDecileScore(Action):
    def __init__(self, features):
        super().__init__(name='ChangeDecileScore',
                         description='change decile score',
                         type='Numeric',
                         features=features,
                         target_features=['decile_score'],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        return self.features['decile_score'].change_feature_value(instance,
                                                                     self.get_param(p,
                                                                                    use_tensor),
                                                                     use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_edu = self.features['decile_score'].get_feature_value(new_instance, use_tensor,
                                                                     space='x')
        old_edu = self.features['decile_score'].get_feature_value(instance, use_tensor, space='x')
        cost = tf.abs(new_edu - old_edu) if use_tensor else np.abs(new_edu - old_edu)

        return self.return_cost(instance, new_instance, cost, use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        new_edu = self.features['decile_score'].get_feature_value(new_instance, use_tensor,
                                                                     space='x')
        old_edu = self.features['decile_score'].get_feature_value(instance, use_tensor,
                                                                     space='x')
        return cond.op_and(cond.op_and(cond.op_gt(new_edu, 0, use_tensor, scale=100.),
                                       cond.op_lt(new_edu, 10, use_tensor, scale=100.),
                                       use_tensor),
                           cond.op_and(cond.op_gt(old_edu, new_edu, use_tensor, scale=10.),
                                       cond.op_gt(new_edu, 0, use_tensor, scale=10.),
                                       use_tensor),
                           use_tensor)


actions = [
    ChangeChargeDegree,
    ChangeDecileScore,
    WaitYears
]
