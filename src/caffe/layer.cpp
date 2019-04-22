#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {

  DLOG(INFO) << "Layer::InitMutex()";
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    DLOG(INFO) << "Layer::Lock()";
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    DLOG(INFO) << "Layer::Unlock()";
    forward_mutex_->unlock();
  }
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
