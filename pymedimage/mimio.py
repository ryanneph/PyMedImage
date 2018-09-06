import os
import re

def contains_dicom(d, recursive=False):
    for (root, dirs, files) in os.walk(d):
        for f in files:
            if os.path.splitext(f)[1].lower() in ['.dcm', '.dicom']:
                return True
        if not recursive:
            break
    return False

def is_mim_doi_dir(d):
    for sub in os.listdir(d):
        fullsub = os.path.join(d, sub)
        if '__Studies' in sub and os.path.isdir(fullsub):
            if contains_dicom(fullsub, recursive=True):
                return True
    return False


def walk_mim_studies(root):
    """generator that yeilds (root, dcmdirs, ) for each path within root recursively"""
    for (root, dirs, files) in os.walk(root):
        remove_subdirs = []
        for sub in dirs:
            fullsub = os.path.join(root, sub)
            if is_mim_doi_dir(fullsub):
                doi = build_doidatacollection(fullsub)
                if doi:
                    remove_subdirs.append(sub)
                    yield doi
        # remove subdirs from further processing
        _moddirs = dirs
        for sub in remove_subdirs:
            _moddirs.remove(sub)
        dirs[:] = _moddirs

def build_doidatacollection(root):
    doi = DOIDataCollection()
    doi.root = root
    doi.id = os.path.basename(root)
    for study_dir in [x for x in os.listdir(root) if '__Studies' in x and os.path.isdir(os.path.join(root, x))]:
        for sub in [x for x in os.listdir(os.path.join(root, study_dir)) if os.path.isdir(os.path.join(root, study_dir, x))]:
            re_formats = [
                # MIM EXPORT FORMAT: "LASTNAME^FIRSTNAME^MIDDLENAME_MODALITY_DATE_MRN_DESCRIPTION..."
                r'^(?P<lastname>[^_]+)[\^_](?P<firstname>[^_]+)(?:\^(?P<middlename>[^_]*))?[\^_](?P<modality>[a-zA-Z0-9 -]*)[\^_](?P<date>(?:[\d]*-?)+)[\^_](?P<mrn>[\d-]*)[\^_](?P<description>.*)$',
                       ]
            for r in re_formats:
                p = re.compile(r, re.IGNORECASE)
                m = p.search(os.path.basename(sub))
                if m is not None:
                    break
            if m is None:
                print('ERROR: no format matched regexp definition for "{!s}"'.format(os.path.join(root, study_dir, sub)))
                continue

            d = {}
            for k, v in m.groupdict().items():
                if v is not None:
                    if isinstance(v, str):
                        v = v.lower()
                    d[k] = v
            d['path'] = sub
            d['fullpath'] = os.path.join(study_dir, sub)

            doi.series.append(d)
    if not doi.series:
        doi = None
    return doi


class DOIDataCollection():
    """POD struct describing a collection of dicom volumes for a patient in the filesystem"""
    def __init__(self):
        self.root = None
        self.id = None
        self.series = []
        self.meta = {}

    def __str__(self):
        return '{!s} => "{!s}"\n'.format(self.id, self.root) + \
               '  series:\n    - {!s}\n'.format('\n    - '.join(['{!s} ({!s})  => "{!s}"'.format(x['date'], x['modality'], x['fullpath']) for x in self.series])) + \
               '  meta:    {!s}'.format(self.meta)

    def getSeriesByModality(self, mod):
        """return list of studies that match the query modality"""
        return [x for x in self.series if x['modality'] == mod.lower()]

    def getRTStructFiles(self):
        """return fullpath for every .dcm file that qualifies as rtstruct"""
        fp = []
        for rtpath in [x['fullpath'] for x in self.getSeriesByModality('rtst')]:
            fp += [os.path.join(rtpath, x) for x in os.listdir(os.path.join(self.root, rtpath)) if '.dcm' == os.path.splitext(x)[1].lower()]
        return fp

    def modalities(self):
        """return list of detected modalities detected in series"""
        mods = set()
        for s in self.series:
            m = s["modality"]
            if isinstance(m, str):
                mods.add(m)
        return mods

