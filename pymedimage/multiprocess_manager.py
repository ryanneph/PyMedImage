import sys
import os
import time
import logging
import multiprocessing
from pymedimage.notifications import pushNotification
from abc import ABCMeta, abstractmethod

# initialize module logger
logger = logging.getLogger(__name__)

class MultiprocessManagerBase:
    """Base class for multiprocessing with standard logging function and notifications"""
    __metaclass__ = ABCMeta
    processes = multiprocessing.cpu_count()

    def __init__(self, title="Generic Multiprocess Manager", processes=processes, notify=True, skip_exceptions=False):
        self.title = title
        self.processes = processes
        self.notify = notify
        self.skip_exceptions = skip_exceptions

    @abstractmethod
    def WorkerFunction(self, argmap):
        """abstract class defining worker function"""
        pass

    def LogStringGenerator(self, worker_results):
        (result_code, result_string, job_time_string, doi) = worker_results
        if isinstance(doi, list): doi = doi[0]
        log_string = '[{string:12s}:{code:2d}]: {doi!s:9s}  {time!s}'.format(
            string  = result_string,
            code    = result_code,
            doi     = doi,
            time    = job_time_string
        )
        return log_string

    def execute(self, argmap):
        """multithreaded manager for calculating local image features for a large number of dois"""
        try:
            # start multiprocessing and print output summary for each job
            time_start = time.time()
            total_jobs = len(argmap)
            jnum = 0
            error_count = 0
            # limit number of concurrent processes as there is only so much GPU memory available at one time
            # with 8 proc: max mem usage of ~4-4.5GB of 12.204GB total global mem
            with multiprocessing.Pool(processes=self.processes) as p:
                logger.info(str(self.title))
                logger.info('-----------------------------------------------------------------------------------------')
                logger.info('BEGINNING PROCESSING (at {!s})'.format(time.strftime('%Y-%b-%d %H:%M:%S')))
                logger.info('')
                logger.info('RESULTS:  (total #jobs: {:d})'.format(total_jobs))
                logger.info('-----------------------------------------------------------------------------------------')

                for worker_results in p.imap(self.WorkerFunction, argmap, chunksize=1):
                    jnum += 1
                    result_code = worker_results[0]
                    log_string = self.LogStringGenerator(worker_results)
                    if log_string:
                        log_string = 'job#{jnum:_>5d} '.format(jnum=jnum) + log_string
                        logger.info(log_string)

                    if (result_code == -1 or (result_code > 0 and result_code < 10)):
                        error_count += 1
                        logger.error('{:05d}.  {!s}'.format(error_count, log_string))

            time_finish = time.time()
            logger.info('-----------------------------------------------------------------------------------------')
            logger.info('(success: {:d} | error: {:d}) of {:d} jobs'.format(total_jobs-error_count, error_count,
                                                                            total_jobs))
            logger.info('total time: {:s}'.format(time.strftime('%H:%M:%S', time.gmtime(time_finish-time_start))))
            logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            logger.info('')

        except Exception as e:
            if self.notify: pushNotification('FAILURE - {!s}'.format(self.title), '{!s}'.format(repr(e)))
            raise e
        if self.notify: pushNotification('SUCCESS - {!s}'.format(self.title), 'Finished processing {:d} jobs \
                                     with {:d} errors'.format(total_jobs, error_count))

    def debug(self, argmap):
        # start multiprocessing and print output summary for each job
        time_start = time.time()
        total_jobs = len(argmap)
        jnum = 0
        error_count = 0

        logger.info(str(self.title))
        logger.info('-----------------------------------------------------------------------------------------')
        logger.info('BEGINNING DEBUG (at {!s})'.format(time.strftime('%Y-%b-%d %H:%M:%S')))
        logger.info('')
        logger.info('RESULTS:  (total #jobs: {:d})'.format(total_jobs))
        logger.info('-----------------------------------------------------------------------------------------')

        for i in range(total_jobs):
            results = self.WorkerFunction(argmap[i])
            jnum += 1
            result_code = results[0]
            log_string = self.LogStringGenerator(results)
            if log_string:
                log_string = 'job#{jnum:_>5d} '.format(jnum=jnum) + log_string
                logger.info(log_string)

            if (result_code == -1 or (result_code > 0 and result_code < 10)):
                error_count += 1
                logger.error('{:05d}.  {!s}'.format(error_count, log_string))

        time_finish = time.time()
        logger.info('-----------------------------------------------------------------------------------------')
        logger.info('(success: {:d} | error: {:d}) of {:d} jobs'.format(total_jobs-error_count, error_count,
                                                                        total_jobs))
        logger.info('total time: {:s}'.format(time.strftime('%H:%M:%S', time.gmtime(time_finish-time_start))))
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logger.info('')
