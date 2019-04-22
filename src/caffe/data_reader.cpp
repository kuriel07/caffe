#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

// It has to explicitly initialize the map<> in order to work. It seems to be a
// gcc bug.
// http://www.cplusplus.com/forum/beginner/31576/
template <>
map<const string, weak_ptr<DataReader<Datum>::Body> >
  DataReader<Datum>::bodies_
  = map<const string, weak_ptr<DataReader<Datum>::Body> >();
template <>
map<const string, weak_ptr<DataReader<AnnotatedDatum>::Body> >
  DataReader<AnnotatedDatum>::bodies_
  = map<const string, weak_ptr<DataReader<AnnotatedDatum>::Body> >();
static boost::mutex bodies_mutex_;

template <typename T>
DataReader<T>::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  printf("DataReader::New lock\n");
  body_ = weak.lock();
  if (!body_) {
    printf("DataReader::New reset\n");
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }

  
  printf("DataReader::New push queue\n");
  body_->new_queue_pairs_.push(queue_pair_);
}

template <typename T>
DataReader<T>::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  printf("DataReader::Delete lock\n");
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    printf("DataReader::Delete erase\n");
    bodies_.erase(key);
  }
  printf("DataReader::Delete done\n");
}

template <typename T>
DataReader<T>::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of data
  printf("DataReader::QueuePair initialize %d\n", size);
  for (int i = 0; i < size; ++i) {
    free_.push(new T());
  }
}

template <typename T>
DataReader<T>::QueuePair::~QueuePair() {
  T* t;
  printf("DataReader::QueuePair free\n");
  while (free_.try_pop(&t)) {
    delete t;
  }
  while (full_.try_pop(&t)) {
    delete t;
  }
  printf("DataReader::QueuePair done\n");
}

template <typename T>
DataReader<T>::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  printf("DataReader::Body start thread\n");
  StartInternalThread();
}

template <typename T>
DataReader<T>::Body::~Body() {
  printf("DataReader::Body stop thread\n");
  StopInternalThread();
  printf("DataReader::Body stop thread done\n");
}

template <typename T>
void DataReader<T>::Body::InternalThreadEntry() {
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  db->Open(param_.data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  vector<shared_ptr<QueuePair> > qps;
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    printf("DataReader::InternalThreadEntry push queue = %d\n", solver_count);
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursor.get(), qp.get());
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
    //while(1) {
      for (int i = 0; i < solver_count; ++i) {
        read_one(cursor.get(), qps[i].get());
        //new_queue_pairs_.push(qps[i]);
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      //printf("DataReader::InternalThreadEntry new_queue_pairs_.size() = %d\n", new_queue_pairs_.size());
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
    printf("DataReader::InternalThreadEntry stopped\n");
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
    printf("DataReader::InternalThreadEntry Interrupted\n");
  }
}

template <typename T>
void DataReader<T>::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  T* t = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  t->ParseFromString(cursor->value());
  qp->full_.push(t);
  //if(t == NULL) LOG(INFO) << "DataReader::read_one (null)";
  //else LOG(INFO) << "DataReader::read_one exist";
  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    LOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

// Instance class
template class DataReader<Datum>;
template class DataReader<AnnotatedDatum>;

}  // namespace caffe
