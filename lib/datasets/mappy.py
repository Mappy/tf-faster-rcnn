import cPickle
import os
import subprocess

import numpy as np
import scipy.io as sio
import scipy.sparse
import pickle
from datasets.imdb import imdb
import uuid
from .mappy_eval import mappy_eval
from model.config import cfg


class mappy(imdb):
    def __init__(self, image_set):
        print("mappy __init__")
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path()
        print("mappy __init__ devkit_path {}".format(self._devkit_path))
        self._data_path = self._devkit_path
        print("mappy __init__ data_path {}".format(self._data_path))

        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        #self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))

        self._image_ext = ['.jpg', '.png']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # Specific config options
        self.config = {'cleanup': True, 'use_salt': True, 'top_k': 2000, 'use_diff': False}

        assert os.path.exists(self._devkit_path), 'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Images', index + ext)
            if os.path.exists(image_path):
                assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
                return image_path

        return False

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        image_set_file = os.path.join(self._data_path, 'ImageSets', self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self, cache=False):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        print 'gt_roidb cache {}'.format(cache)
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if cache:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    roidb = cPickle.load(fid)
                print '{} gt roidb loaded from {}'.format(self.name, cache_file)
                return roidb

        gt_roidb = [self._load_mappy_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self, cache=False):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        print("selective_search_roidb cache= %s" % (cache))

        cache_file = os.path.join(self.cache_path, self.name + '_selective_search_roidb.pkl')
        if cache:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    roidb = cPickle.load(fid)
                print '{} ss roidb loaded from {}'.format(self.name, cache_file)
                return roidb
        print("selective_search_roidb self._image_set= %s" % (self._image_set))
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        print("selective_search_roidb roidb= %s" % (roidb))

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        print("mappy _load_selective_search_roidb >> ")

        filename = os.path.abspath(os.path.join(self._devkit_path,
                                                self.name + '.mat'))
        assert os.path.exists(filename), 'Selective search data not found at: {}'.format(filename)
        mat = sio.loadmat(filename)
        raw_data = mat["all_boxes"]

        box_list = [raw_data[i][:, (1, 0, 3, 2)] for i in range(raw_data.shape[0])]

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        print("mappy _load_selective_search_IJCV_roidb >> ")
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        assert os.path.exists(IJCV_path), 'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :] - 1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        print("selective_search_IJCV_roidb >> ")

        cache_file = os.path.join(self.cache_path,
                                  '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                                  format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            print("selective_search_IJCV_roidb exists(cache_file) ")
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        print("selective_search_IJCV_roidb OK ")

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_mappy_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of INRIAPerson.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        with open(filename) as f:
            data = f.read()
        import re
        objs = re.findall('\(\d+(?:\.\d+)?, \d+(?:\.\d+)?, \d+(?:\.\d+)?, \d+(?:\.\d+)?, \d\)', data)
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            coor = re.findall('\d+(?:\.\d+)?', obj)
            x1 = float(coor[0])
            y1 = float(coor[1])
            w = float(coor[2])
            h = float(coor[3])
            cls = int(coor[4])

            x2 = x1 + w
            y2 = y1 + h

            if cls == 1:
                cls = 15    #person
            elif cls == 2:
                cls = 7     #car

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def _write_inria_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format("matlab")
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); mappy_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
            .format(self._devkit_path, comp_id,
                    self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def _write_mappy_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} Mappy results file'.format(cls))
            filename = self._get_mappy_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_mappy_results_file_template(self):
        # mappy/results/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'Annotations',
            '{:s}.txt')
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True #if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_mappy_results_file_template().format(cls)
            rec, prec, ap = mappy_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        print('evaluate_detections output_dir='+output_dir)
        self._write_mappy_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def evaluate_detections_with_matlab(self, all_boxes, output_dir):
        comp_id = self._write_inria_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'mappy')
