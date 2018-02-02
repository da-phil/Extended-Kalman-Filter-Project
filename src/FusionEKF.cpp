#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <string>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;
  timestep_ = 0;
  previous_timestamp_ = 0;
  noise_ax = 9;
  noise_ay = 9;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_      = MatrixXd(3, 4);
  MatrixXd Q_ = MatrixXd(4, 4);
  MatrixXd P_ = MatrixXd(4, 4);
  VectorXd x_ = VectorXd::Zero(4);

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */

  //measurement covariance matrix - laser
  R_laser_ << 0.0225,     0,
              0,          0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09,   0,        0,
              0,      0.0009,   0,
              0,      0,        0.09;

  //measurement matricies
  H_laser_ << 1,    0,    0,    0,
              0,    1,    0,    0;

  //state covariance matrix P
  P_ << 1,    0,    0,      0,
        0,    1,    0,      0,
        0,    0,    1000,   0,
        0,    0,    0,      1000;

  //the initial transition matrix F_
  MatrixXd F_ = MatrixXd::Identity(4, 4);

  ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  float dt;
 
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF Init" << endl;
    ekf_.x_ = VectorXd::Zero(4);
    previous_timestamp_ = measurement_pack.timestamp_;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      cout << "Initial radar measurement received!" << endl;
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float rhodot = measurement_pack.raw_measurements_[2];

      ekf_.x_(0) = rho * cos(phi);
      ekf_.x_(1) = rho * sin(phi);
      ekf_.x_(2) = 0.0; //rhodot * cos(phi);
      ekf_.x_(3) = 0.0; //rhodot * sin(phi);

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      cout << "Initial lidar measurement received!" << endl;
      float px, py, vx, vy;
      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];
      vx = 0.0; //measurement_pack.raw_measurements_[0];
      vy = 0.0; //measurement_pack.raw_measurements_[1];
      ekf_.x_ << px, py, vx, vy;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;

  } else {

    timestep_++;

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
     TODO:
       * Update the state transition matrix F according to the new elapsed time.
        - Time is measured in seconds.
       * Update the process noise covariance matrix.
       * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */
    
    //compute the time elapsed between the current and previous measurements
    dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;
    
    //1. Modify the F matrix so that the time is integrated
    // kf_.F_ <<  1, 0, dt, 0,
    //            0, 1, 0,  dt,
    //            0, 0, 1,  0,
    //            0, 0, 0,  1;
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    //2. Set the process noise covariance matrix Q
    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_3 * dt;
    ekf_.Q_ <<  dt_4/4*noise_ax, 0,               dt_3/2*noise_ax,  0,
                0,               dt_4/4*noise_ay, 0,                dt_3/2*noise_ay,
                dt_3/2*noise_ax, 0,               dt_2*noise_ax,    0,
                0,               dt_3/2*noise_ay, 0,                dt_2*noise_ay;

    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
     TODO:
       * Use the sensor type to perform the update step.
       * Update the state and covariance matrices.
     */
    cout << "Timestep " << timestep_ << " - dt: " << dt << " - ";

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      cout << "Radar measurement received!" << endl;
      Hj_ = tools.CalculateJacobian(ekf_.x_);
      // don't update measurement if we can't compute the jacobian
      if (Hj_.isZero(0)){
        cerr << "Hj is zero" << endl;
        return;
      }
      // set H_ to Hj when updating with a radar measurement
      ekf_.H_ = Hj_;
      ekf_.R_ = R_radar_;
      // Radar updates (non-linear model, needs linearization, hence call to UpdateEKF)
      ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
      cout << "Lidar measurement received!" << endl;
      ekf_.H_ = H_laser_;
      ekf_.R_ = R_laser_;
      // Laser updates (linear model)
      ekf_.Update(measurement_pack.raw_measurements_);
    }
  }

  // print the output
  std::string sep = "\n----------------------------------------\n";
  Eigen::IOFormat CleanFmt(cout.precision(3), 0, ", ", "\n", "  [", "]");
  cout << "x_ = " << endl << ekf_.x_.format(CleanFmt) << endl;
  cout << "P_ = " << endl << ekf_.P_.format(CleanFmt) << endl << endl;
}
