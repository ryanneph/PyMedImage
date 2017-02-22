QMODE_FIXEDHU = 0
QMODE_STAT = 1

def enforceGLCMQuantizationMode(feature_def, modality):
    # force stat based GLCM quantization if not CT image
    if 'gray_levels' in feature_def.args and 'binwidth' in feature_def.args:
        if modality and modality.lower() == 'ct':
            # enforce glcm FIXEDHU based quantization
            del feature_def.args['gray_levels']
        else:
            # force glcm STAT based quantization
            del feature_def.args['binwidth']
    # elif 'binwidth' in feature_def.args and modality and modality.lower() != 'ct':
    #     # force glcm STAT based quantization
    #     del feature_def.args['binwidth']
