#undef EIGEN_NO_DEBUG
#define TAPKEE_USE_LGPL_COVERTREE
#define TAPKEE_WITH_ARPACK
#include <tapkee/tapkee.hpp>
#include <tapkee/callbacks/dummy_callbacks.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>
#include <Eigen/Core>

using namespace std;
using namespace tapkee;

#define M 1
#define NB_STEP 50
#define DT 1
#define FMAX 200


struct MyDistanceCallback
{
  double distance(const Eigen::VectorXd& l,const Eigen::VectorXd& r) 
	{ 
	  return (l-r).norm();
	} 

  double kernel(const Eigen::VectorXd& l,const Eigen::VectorXd& r) 
	{ 
	  return (l-r).norm();
	} 
}; 


void simulate(double F, double theta, std::vector<Eigen::VectorXd>& traj)
{

  Eigen::Vector2d a;
  a(0)=F*cos(theta)/M;
  a(1)=(F*sin(theta)-9.81)/M;

  Eigen::Vector2d v;
  v(0)=0;
  v(1)=0;
  Eigen::Vector2d p;
  p(0)=0;
  p(1)=0;
  traj.clear();
  traj.push_back(p);

  for(size_t t=0;t<NB_STEP-1;t++)
    {

      v=v+a*DT;
      p=p+v*DT;
      a(0)=0;
      a(1)=-9.81;

      if(p(1)<=0)//contact with the ground
	{

	  p(1)=0;
	  a(1)=-0.6*v(1); //dumping factor
	  v(1)=0;

	}

      traj.push_back(p);

    }

}



void gen_dataset(Eigen::MatrixXd& data)
{
  size_t reso=100;
  data=Eigen::MatrixXd(reso*reso,100);
  size_t index=0;
  for(size_t i=0;i<reso;i++)
    for(size_t j=0;j<reso;j++)
      {
	std::vector<Eigen::VectorXd> traj;
	simulate(i/(float)reso*200.0 , j/(float)reso*M_PI/2, traj);
	for(size_t t=0; t<traj.size(); t++)
	  {
	    data(index,t)=traj[t][0];
	    data(index,t+50)=traj[t][1];
	  }
	index++;
      }


}

int main(int argc, const char** argv)
{
  tapkee::LoggingSingleton::instance().enable_info();
  tapkee::LoggingSingleton::instance().enable_benchmark();
  tapkee::LoggingSingleton::instance().enable_debug();
  tapkee::ParametersSet parameters = 
    tapkee::kwargs[
		   tapkee::method = tapkee::PCA,
		   tapkee::computation_strategy = tapkee::HomogeneousCPUStrategy,
		   tapkee::eigen_method = tapkee::Arpack,
		   tapkee::neighbors_method = tapkee::CoverTree,
		   tapkee::num_neighbors = 150,
		   tapkee::target_dimension = 2,
		   tapkee::diffusion_map_timesteps = 1,
		   tapkee::gaussian_kernel_width = 1,
		   tapkee::max_iteration = 1000,
		   tapkee::spe_global_strategy = false,
		   tapkee::spe_num_updates = 100,
		   tapkee::spe_tolerance = 1e-5,
		   tapkee::landmark_ratio = 0,
		   tapkee::nullspace_shift = 1e-9,
		   tapkee::check_connectivity = true,
		   tapkee::fa_epsilon = 1e-5,
		   tapkee::sne_perplexity =  30.0,
		   tapkee::sne_theta = 0.5,
		   tapkee::squishing_rate = 0.99];
  
  


  Eigen::MatrixXd data;
  gen_dataset(data);


  std::ofstream file("data.dat");
  file<<data<<std::endl;

  data.transposeInPlace();

  std::ofstream file33("data_after.dat");
  file33<<data<<std::endl;
  
  std::cout<<"dataset loaded"<<std::endl;
  std::cout<< "Data contains " << data.cols() << " feature vectors with dimension of " << data.rows()<<std::endl;
  
  TapkeeOutput output = initialize() 
    .withParameters(parameters).embedUsing(data);
  

  std::ofstream file2("embeding.dat");
  file2<<output.embedding.transpose()<<std::endl;

  std::ofstream file3("projection.dat");
  for(size_t i=0; i<data.cols(); i++)
    file3<<output.projection(data.col(i)).transpose()<< "  "<< data.col(i).transpose()<<std::endl;
  
  
  //  cout << output.embedding.transpose() << endl;
  
  return 0;
}
