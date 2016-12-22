"""data_structures.py

Contains class definitions for inheritence in dataset specific implementations
"""
import os
import logging
import math
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

# initialize module logger
logger = logging.getLogger(__name__)

class DOIBase:
    """Defines a digital object identifier that supplies the paths and filenames for various feature/cluster algorithms"""
    __metaclass__ = ABCMeta
    def __str__(self):
        return str(self.doi)

    def __init__(self, doi):
        self.doi = doi

    @abstractmethod
    def getDicomDataPath(self):
        pass

    @abstractmethod
    def getImageVolume(self):
        pass

    @abstractmethod
    def getROI(self):
        pass

    @abstractmethod
    def getFeaturesPath(self):
        pass

    @abstractmethod
    def getClusterL1PicklePath(self):
        pass

    @abstractmethod
    def getClusterL2PicklePath(self):
        pass


class WritableFeatureDefinition:
    def __init__(self, label, recalculate=False):
        self.label = label
        self.args = OrderedDict()
        self.recalculate = recalculate

    def addArg(self, key, value):
        """abstracts away the complexities of initializing an ordereddict with lists of tuples..."""
        self.args[key] = value

    def generateFilename(self, ignore_list=['glcm_stat_function']):
        return 'feature={label!s}_args=({args!s}).pickle'.format(
                label=self.label,
                args=self.getArgsString(ignore_list)
            )

    def getKeywords(self):
        """generates keywords list for finding a pickled feature file using findFiles()

        Returns:
            list of keyword strings which must be in filename for match
        """
        keywords = ['feature={!s}_'.format(self.label)] + \
                   ['{argname!s}={argval!s}'.format(argname=n, argval=v)
                    for (n, v) in self.args.items()]
        for item in list(keywords):
            if ('function' in item.lower() or 'kernel' in item.lower()):
                keywords.remove(item)
        return keywords

    def getArgsString(self, ignore_list=[]):
        """create standardized arg string based on feature args
        Args:
            args -- ordered dict of argname: argvalue pairs
        Returns:
            string
        """
        args_string_list = []
        for k, v in self.args.items():
            ignore = False
            for ign in ignore_list:
                if (ign.lower() in k.lower()):
                    ignore = True
                    break
            if (ignore):
                continue

            if (callable(v) or 'function' in k.lower()):
                args_string_list.append('function={!s}'.format(k))
            elif ('kernel' in k.lower()):
                continue
            elif isinstance(v, int):
                args_string_list.append('{!s}={:d}'.format(k, v))
            elif isinstance(v, float):
                args_string_list.append('{!s}={:0.2f}'.format(k, v))
            else:
                args_string_list.append('{!s}={!s}'.format(k, v))
        return ','.join(args_string_list)

    def findFiles(self, root, ext='.pickle', casesensitive=False, recursive=False):
        """returns a list of full file paths beneath root if each path contains all of the strings in keywordlist
        and is of the type (ext) specified

        Args:
            root          -- path within which to check files

        Optional Args:
            ext          -- file extension to verify (with or without dot is okay)
            casesensitive -- check character case?
            recursive     -- walk into subdirectories
        """
        # get list of files in root (match extension if specified) ##UPDATE FOR RECURSIVITY
        files_list = []
        for head, dirs, files in os.walk(root, topdown=True):
            for file in files:
                if (ext.replace('.', '') == os.path.splitext(file)[1].replace('.', '')):
                    files_list.append(os.path.join(head, file))
            if not recursive: del dirs[:]

        matches = []
        if (files_list is not None and len(files_list) > 0):
            # find files that contain all specified keywords
            for f in files_list:
                valid = True
                for key in self.getKeywords():
                    if (casesensitive):
                        if (key not in f):
                            valid = False
                            break
                    else:
                        if (key.lower() not in f.lower()):
                            valid = False
                            break
                if (valid):
                    matches.append(os.path.join(root, f))

            # print results to debug
            if (len(matches) == 1):
                logger.debug('match found at path: {:s}'.format(os.path.join(root, matches[0])))
            elif (len(matches) > 1):
                logger.debug('matches found at paths:')
                for m in matches:
                    logger.debug('  {:s}'.format(m))
            else:
                logger.debug('no matches found')
                return None

            return matches
        else:
            logger.debug('no files found')
            return None

class LocalFeatureDefinition(WritableFeatureDefinition):
    """Standard feature definition for use in scripts. provides plug-in ensuring consistent definition"""
    def __init__(self, label, calculation_function, recalculate=False):
        """
        Args:
            label -- string
            calculation_function -- function pointer for patch based local feature calcuation matching
                                    signature: calculation_function(ndArray)
        Optional Args:
            recalculate -- parsed arg or bool literal indicating if feature should be recalculated and pickled
            args -- dict of arg key:value pairs where key exactly matches the kwarg key for calculation_function
        """
        super().__init__(label, recalculate)
        self.calculation_function = calculation_function


class LocalFeatureCompositionDefinition(WritableFeatureDefinition):
    """Used to define a composite of a list of functions to be applied at each patch location within a full image iterator"""
    def __init__(self, collection_label, composition_function, recalculate=False):
        """
        Args:
            collection_label -- string
            composition_function -- function pointer for composing a list of local feature calcuation results
                as defined by the member feature_defs
        Optional Args:
            recalculate -- parsed arg or bool literal indicating if feature should be recalculated and pickled
        """
        super().__init__(collection_label, recalculate)
        self.composition_function = composition_function
        self.featdefs = []

    def addLocalFeatureDefinition(self, featdef):
        """abstracts away the complexities of initializing an ordereddict with lists of tuples..."""
        self.featdefs.append(featdef)

class FeatureList():
    def __init__(self, feature_def_list=None):
        if (feature_def_list and isinstance(feature_def_list, list)):
            self.__storage__ = feature_def_list
        else:
            self.__storage__ = []

    def append(self, item):
        self.__storage__.append(item)

    def __findbylabel__(self, key):
        for item in self.__storage__:
            label = item.label
            if (label == key):
                return item
        raise KeyError

    def __getitem__(self, key):
        nitems = len(self.__storage__)
        if (isinstance(key, str)):
            return self.__findbylabel__(key)

        if (key >= nitems):
            raise IndexError
        elif (key < 0):
            if (math.abs(key) > nitems):
                raise IndexError
            else:
                key = nitems + key
        return self.__storage__.__getitem__(key)

    def __len__(self):
        return len(self.__storage__)

    def __iter__(self):
        for element in self.__storage__:
            yield element


