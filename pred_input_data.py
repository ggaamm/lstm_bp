from __future__ import print_function
import gzip
import os
import urllib
import numpy
#SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

class DataSet(object):
  def __init__(self,input_trace,result_trace):

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    self._num_examples = input_trace.shape[0]

    self._input_traces = input_trace
    self._result_traces = result_trace
    self._trace_length = input_trace.shape[0]
    self._index_in_trace = -1

  @property
  def input_traces(self):
    return self._input_traces

  @property
  def result_traces(self):
    return self._result_traces

  @property
  def input_traces(self):
    return self._input_traces

  @property
  def result_traces(self):
    return self._result_traces

  @property
  def num_examples(self):
    return self._trace_length

  @property
  def epochs_completed(self):
    return self._index_in_trace

  def next_batch(self, batch_size, timesteps=1, shift=None):
    batch_array_input = numpy.array([]).reshape(0,self._input_traces.shape[1])
    batch_array_result = numpy.array([]).reshape(0,self._input_traces.shape[1])
    for i in xrange(batch_size):
      startb = self._index_in_trace + 1
      if shift is None:
        endb = startb + timesteps
      elif shift is not None and self._index_in_trace >= timesteps:
        endb = startb + timesteps
      else:
        startb = 0
      endb = startb + timesteps
      self._index_in_trace = endb
      #Finished trace get the next trace available
      #Get the following trace start from zero
      if self._index_in_trace > self._trace_length:
        batch_array_input = numpy.vstack([batch_array_input,self._input_traces[0:timesteps]])
        batch_array_result = numpy.vstack([batch_array_result,self._result_traces[timesteps-1:timesteps]])
      else:
        batch_array_input = numpy.vstack([batch_array_input,self._input_traces[startb:endb]])
        batch_array_result = numpy.vstack([batch_array_result,self._result_traces[endb-1:endb]])

    return batch_array_input,batch_array_result





def read_data_sets(train_dir, test_dir=None, one_hot=False):
  class DataSets(object):
    pass

  train_trace = numpy.loadtxt(train_dir, delimiter=' ', usecols=[1],converters={1: lambda s: numpy.array([1.,0.]) if float(s)<1 else  numpy.array([0.,1.])}) #use first column
  train_res_trace = numpy.roll(train_trace, 2)  ##shift train_pred one to right, 2 for dual column
  #train_trace = numpy.loadtxt(train_dir, delimiter=' ', usecols=[1]) #use first column
  #for val in range(0, 20):
   # print("{0} {1}".format(train_trace[val],train_res_trace[val]))
  #test_trace = numpy.loadtxt(test_dir, delimiter=' ', usecols=[1]) #use first column
  data_sets = DataSets()

  VALIDATION_SIZE = 1000
  train_traces = train_trace[VALIDATION_SIZE:]
  train_results = train_res_trace[VALIDATION_SIZE:]
  validation_traces = train_trace[:VALIDATION_SIZE]
  validation_results = train_res_trace[:VALIDATION_SIZE]

  if test_dir is not None:
    '''
    test_trace = numpy.loadtxt(test_dir, delimiter=' ', usecols=[1], converters={
      1: lambda s: numpy.array([1., 0.]) if float(s) < 1 else  numpy.array([0., 1.])})  # use first column
    test_result = numpy.roll(test_trace, 2)  ##shift train_pred one to right, 2 for dual column
    '''
    test_trace = numpy.loadtxt(test_dir, delimiter=' ', usecols=[1],converters={1: lambda s: numpy.array([1.,0.]) if float(s)<1 else  numpy.array([0.,1.])})
    test_result = numpy.roll(test_trace, 2)  ##shift train_pred one to right, 2 for dual column

    data_sets.validation = DataSet(validation_traces, validation_results)
    data_sets.test = DataSet(test_trace, test_result)
    print(test_result.shape)

  data_sets.train = DataSet(train_trace, train_res_trace)
  data_sets.test = DataSet(train_traces, train_res_trace)
  #print(train_traces.shape)

  return data_sets


