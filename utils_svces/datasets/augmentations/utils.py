import PIL.Image as Img

INTERPOLATION_STRING_TO_TYPE = {
    'nearest': Img.NEAREST,
    'bilinear': Img.BILINEAR,
    'bicubic': Img.BICUBIC,
    'lanczos': Img.LANCZOS
                                }