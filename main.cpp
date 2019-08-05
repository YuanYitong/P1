#include <iostream>
#include <vector>
#include <cmath>
#include "ceres/ceres.h"
#include "glog/logging.h"

using namespace std;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

const double optimize_u_resolution = 5.0;
const double refine_resolution = 0.01;

const double roi_begin = 5.0;
const double roi_end = 50.0;

const double angle_upper_bound = 0.0;
const double angle_lower_bound = 21.0;
const double angle_step = 1.0;

const double h = 2.0;
int beams = 16;

const double u_coe1 = 5;
const double u_coe2 = 50;
const double s_coe = 20;

const double factor1 = 1.0;
const double factor2 = 5.0;


struct F_con_Residual{
    F_con_Residual(double u): u_(u) {}

    double Coe1(double u)const{
        return pow((1.0 * (u - u_coe1) / s_coe) , 2)  + 1;
    }

    double Coe2(double u)const{
        return pow((1.0 * (u - u_coe2) / s_coe) , 2)  + 1;
    }

    template <typename T> void I(double u, const T c[], T c2[])const{
        int j = 0;
        for(int i = 0; i< beams; i++){
            if(u * tan(c[i] * T(M_PI / 180)) < h){
                c2[j] = c[i];
                j++;
            }
        }
    }

    template <typename T> T con(double u, const T c[])const{
        T count = T(0);
        T c_[beams];
        I(u_,c,c_);
        for(int i = 0;i < beams; i++){
            if(c_[i]!= T(0)){
                count += T(1);
            }
        }
        return count;
    }

    template <typename T> T dis(double u, const T c[])const{
        T count = T(0);
        T c_[beams];
        I(u_,c,c_);
        for(int i = 0;i < beams; i++){
            if(c_[i]!= T(0)){
                count += T(1);
            }
        }

        if(count == T(0))  return T(0);
        else{
            T c_max = T(0);
            T c_min = T(0);
            c_max = c_min = c_[0];

            for(int i = 1;i <beams; i++){
                if(c_[i] > c_max)  c_max = c_[i];
                else if(c_[i] < c_min) c_min = c_[i];
            }

            return T(1.0 * (c_max - c_min));
        }
    }

    template <typename T> bool operator()(const T* const c,
                                            T* residual) const{
        residual[0] = factor1 * Coe1(u_) * dis<T>(u_,c) + factor2 * Coe2(u_) * con<T>(u_,c);
        return true;
    }


private:
    const double u_;
};

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);

    double c[16] = {0,1.5,2,3,4,5,6,7,8,9,10,11,12,13,14,29.01};

    Problem problem;
    for(int i = 0;i < (roi_end - roi_begin) / optimize_u_resolution; ++i) {
        problem.AddResidualBlock(
                new AutoDiffCostFunction<F_con_Residual, 1, 16>(
                        new F_con_Residual(roi_begin + i * optimize_u_resolution)),
                NULL,
                c);
    }

    Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options,&problem,&summary);
    std::cout<<summary.BriefReport()<< "\n";
    for(int i = 0; i< 16;i++){
        cout<<"i = "<<i<<"    "<<c[i]<<endl;
    }

    return 0;
}