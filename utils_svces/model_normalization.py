import torch

class NormalizationWrapper(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x, *args, **kwargs):
        x_normalized = (x - self.mean)/self.std
        return self.model(x_normalized, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict()

def IdentityWrapper(model):
    mean = torch.tensor([0., 0., 0.])
    std = torch.tensor([1., 1., 1.])
    return NormalizationWrapper(model, mean, std)

def Cifar10Wrapper(model):
    mean = torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
    std = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])
    return NormalizationWrapper(model, mean, std)

def Cifar100Wrapper(model):
    mean = torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
    std = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])
    return NormalizationWrapper(model, mean, std)

def SVHNWrapper(model):
    mean = torch.tensor([0.4377, 0.4438, 0.4728])
    std = torch.tensor([0.1201, 0.1231, 0.1052])
    return NormalizationWrapper(model, mean, std)

def CelebAWrapper(model):
    mean = torch.tensor([0.5063, 0.4258, 0.3832])
    std = torch.tensor([0.2632, 0.2424, 0.2385])
    return NormalizationWrapper(model, mean, std)

def TinyImageNetWrapper(model):
    mean = torch.tensor([0.4802, 0.4481, 0.3975])
    std = torch.tensor([0.2302, 0.2265, 0.2262])
    return NormalizationWrapper(model, mean, std)

def ImageNetWrapper(model):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return NormalizationWrapper(model, mean, std)

def RestrictedImageNetWrapper(model):
    mean = torch.tensor([0.4717, 0.4499, 0.3837])
    std = torch.tensor([0.2600, 0.2516, 0.2575])
    return NormalizationWrapper(model, mean, std)

def BigTransferWrapper(model):
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    return NormalizationWrapper(model, mean, std)

def LSUNScenesWrapper(model):
    #imagenet
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return NormalizationWrapper(model, mean, std)


def FundusKaggleWrapper(model):
    # using distance_check.py (for not background-subtracted!)
    mean = torch.tensor([0.4509289264678955, 0.29734623432159424, 0.20647032558918])
    std = torch.tensor([0.27998650074005127, 0.191138356924057, 0.1482602059841156])
    return NormalizationWrapper(model, mean, std)

def FundusKaggleWrapper_raw_v2(model):
    # using distance_check.py (for not background-subtracted!)
    mean = torch.tensor([0.4136027693748474, 0.26003414392471313, 0.17126449942588806]) #[0.4136027693748474, 0.26003414392471313, 0.17126449942588806])
    std = torch.tensor([0.2892555594444275, 0.1874758005142212, 0.1358364075422287])
    return NormalizationWrapper(model, mean, std)

def FundusKaggleWrapper_raw_v2_new_qual_eval(model):
    # using distance_check.py (for not background-subtracted!)
    mean = torch.tensor([0.4129297733306885, 0.2720150351524353, 0.1855563372373581]) #[0.4136027693748474, 0.26003414392471313, 0.17126449942588806])
    std = torch.tensor([0.2923908531665802, 0.19916436076164246, 0.15385839343070984])
    return NormalizationWrapper(model, mean, std)

def FundusKaggleWrapper_raw_v2_new_qual_eval_artifacts_green_circles_blue_squares(model):
    mean = torch.tensor([0.4090270400047302, 0.27157893776893616, 0.1879388391971588])
    std = torch.tensor([0.2932538390159607, 0.20203307271003723, 0.16202093660831451])
    return NormalizationWrapper(model, mean, std)



def FundusKaggleWrapper_clahe_v2_new_qual_eval(model):
    mean = torch.tensor([0.4502241015434265, 0.3076121509075165, 0.22690969705581665])
    std = torch.tensor([0.302660197019577, 0.20975057780742645, 0.16311487555503845])
    return NormalizationWrapper(model, mean, std)

def FundusKaggleWrapper_clahe_v2_new_qual_eval_drop1(model):
    mean = torch.tensor([0.41859379410743713, 0.28517860174179077, 0.2082202434539795])
    std = torch.tensor([0.30145615339279175, 0.21559025347232819, 0.16515834629535675])
    return NormalizationWrapper(model, mean, std)


def FundusKaggleWrapper_raw_clahe_v2(model):
    # using distance_check.py (for not background-subtracted!)
    mean = torch.tensor([0.44815593957901, 0.29301443696022034, 0.2084226906299591])#[0.44340115785598755, 0.2918033003807068, 0.21094724535942078])
    std = torch.tensor([0.29835382103919983, 0.1983824223279953, 0.1487717479467392])#[0.3023217022418976, 0.20483310520648956, 0.15324123203754425])
    return NormalizationWrapper(model, mean, std)

def FundusKaggleWrapper_background_subtracted_v2(model):
    # Computed for background subtracted with median filter, v2
    mean = torch.tensor([0.5065840482711792, 0.5020201206207275, 0.5052798986434937])
    std = torch.tensor([0.08260940760374069, 0.09632422775030136, 0.06450627744197845])
    return NormalizationWrapper(model, mean, std)

def FundusKaggleWrapper_background_subtracted_gauss_v2(model):
    # Computed for background subtracted with median filter, v2
    mean = torch.tensor([0.5031158924102783, 0.5022450089454651, 0.5019901990890503])
    std = torch.tensor([0.06590327620506287, 0.07294944673776627, 0.046418510377407074])
    return NormalizationWrapper(model, mean, std)

def FundusKaggleWrapper_background_subtracted_gauss_v2_masked(model):
    # Computed for background subtracted with median filter, v2
    mean = torch.tensor([0.504446804523468, 0.5120429992675781, 0.5033383369445801])
    std = torch.tensor([0.18714767694473267, 0.1957288533449173, 0.15363547205924988])
    return NormalizationWrapper(model, mean, std)

def FundusKaggleWrapperBackgroundSubtracted(model):
    # using distance_check.py (for background-subtracted!)
    # Note: the first models were run with FundusKaggleWrapper!
    mean = torch.tensor([0.5038442015647888, 0.5026502013206482, 0.5020962357521057])
    std = torch.tensor([0.06049098074436188, 0.07234558463096619, 0.045092713087797165])
    return NormalizationWrapper(model, mean, std)

def OCTWrapper_first1000(model):
    mean = torch.tensor([0.1950831264257431])#, 0.1950831264257431, 0.1950831264257431])
    std = torch.tensor([0.22118829190731049])#, 0.22118829190731049, 0.22118829190731049])
    return NormalizationWrapper(model, mean, std)
