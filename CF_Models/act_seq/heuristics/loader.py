from CF_Models.act_seq.heuristics.vanilla import VanillaHeuristics
from CF_Models.act_seq.heuristics.step_full import StepFullHeuristics
from CF_Models.act_seq.heuristics.abs_gradient import AbsGradHeuristics
from CF_Models.act_seq.heuristics.quickdraw import QuickDrawDriver


def load_heuristics(type, actions, model, length=4):
    if type == 'vanilla':
        return VanillaHeuristics(actions, max_length=length)
    elif type == 'step-full':
        return StepFullHeuristics(actions, max_length=length)
    elif type == 'abs_grad':
        return AbsGradHeuristics(actions, model, max_length=length)
    elif type == 'quickdraw':
        return QuickDrawDriver(actions)
    else:
        raise ValueError('No matching heuristics')
