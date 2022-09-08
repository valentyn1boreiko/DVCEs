from .adversarialattack import AdversarialAttack

##################################
class DummyAttack(AdversarialAttack):
    def __init__(self):
        super().__init__(None, 0, model=None)

    def perturb(self, x, y, targeted=False, x_init=None):
        return x
