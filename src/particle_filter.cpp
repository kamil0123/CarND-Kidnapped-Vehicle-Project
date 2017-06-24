/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// http://www.cplusplus.com/reference/random/default_random_engine/
// random number engine class that generates pseudo-random numbers
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// number of particles
	num_particles = 100;

	// noise 
	normal_distribution<double> noise_x(0, std[0]);
	normal_distribution<double> noise_y(0, std[1]);
	normal_distribution<double> noise_theta(0, std[2]);

	double weight = 1.0/num_particles;

	// init particles
	for (int i = 0; i < num_particles; i++) {
		Particle particle;

		particle.id = i;
		particle.x = x;
		particle.y = y;
		particle.theta = theta;
		particle.weight = weight;

		// add noise
		particle.x += noise_x(gen);
		particle.y += noise_y(gen);
		particle.theta += noise_theta(gen);

		particles.push_back(particle);
		weights.push_back(weight);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// noise 
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	// add measurements to each particle particles
	for (int i = 0; i < num_particles; i++) {
		
		if (fabs(yaw_rate) < 0.0001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// add noise
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
		particles[i].theta += noise_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// predicted - landmarks which are predicted to be in sensor range of particle
	// observations - observations (measurements) of car in map coordinate system

	if (observations.size()>0) {

		for (int i = 0; i <observations.size(); i++) {
			// observations[i] - current observation

			int best_prediction_id = predicted[0].id;
			double distance_x = observations[i].x - predicted[0].x;
			double distance_y = observations[i].y - predicted[0].y;
			double min_distance = sqrt(distance_x * distance_x + distance_y * distance_y);
			double new_distance = 0.0;

			for (int j = 1; j < predicted.size(); j++) {
				distance_x = observations[i].x - predicted[j].x;
				distance_y = observations[i].y - predicted[j].y;
				new_distance = sqrt(distance_x * distance_x + distance_y * distance_y);

				if (new_distance < min_distance) {
					min_distance = new_distance;
					best_prediction_id = predicted[j].id;
				}
			}
			observations[i].id = best_prediction_id;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
		for (int i = 0; i < num_particles; i++) {
		
		// landmarks which are predicted to be in sensor range of particle
		vector<LandmarkObs> predicted;
		// observations (measurements) of particles in map coordinate system (after transformation from car coordinate system)
		vector<LandmarkObs> transformed_observations;
		
		Particle particle = particles[i];
		
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			
			// get single landmark data 
			int   landmark_id = map_landmarks.landmark_list[j].id_i;
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			
			double distance_x = landmark_x - particle.x;
			double distance_y = landmark_y - particle.y;
			double distance = sqrt(distance_x * distance_x + distance_y * distance_y);
			
			if (fabs(distance) <= sensor_range) {
				predicted.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
			}
		}
		
		// http://planning.cs.uiuc.edu/node99.html
		// equation 3.33
		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs delta_pos = observations[j];
			double transformed_obs_x = delta_pos.x * cos(particle.theta) - delta_pos.y * sin(particle.theta) + particle.x;
			double transformed_obs_y = delta_pos.x * sin(particle.theta) + delta_pos.y * cos(particle.theta) + particle.y;
			
			transformed_observations.push_back(LandmarkObs{observations[j].id, transformed_obs_x, transformed_obs_y});
		}

		// data association:
		// each landmark m need to be associated with nearest landmark observation in map coordinate system
		dataAssociation(predicted, transformed_observations);
		// restart weight
		particle.weight = 1.0;

		for (int j = 0; j < transformed_observations.size(); j++) {
			// get observed postion of particle
			LandmarkObs transformed_observation = transformed_observations[i];
			int transformed_obs_id = transformed_observation.id;
			double transformed_obs_x = transformed_observation.x;
			double transformed_obs_y = transformed_observation.y;

			// position of associated landmark
			double predicted_x;
			double predicted_y;

			for (int k = 0; k < predicted.size(); k++) {
				if (predicted[k].id == transformed_obs_id) {
					predicted_x = predicted[k].x;
					predicted_y = predicted[k].y;
				}
			}

			// calculate weight for current particle
			double particle_weight;
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
		
			double factor = 1.0 / (2.0 * M_PI * std_x * std_y);
			double weight_x_part = pow(transformed_obs_x - predicted_x,2)  / (2 * pow(std_x, 2));
			double weight_y_part = pow(transformed_obs_y - predicted_y,2)  / (2 * pow(std_y, 2));

			particle_weight = factor * exp(-(weight_x_part + weight_y_part));

			// total observations weight
			particles[i].weight *= particle_weight;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// method of resampling: resampling wheel:
	// https://www.youtube.com/watch?v=wNQVo6uOgYA
	// http://calebmadrigal.com/resampling-wheel-algorithm/

	vector<Particle> new_particles;
	double max_weight;
	double beta = 0.0;

	uniform_real_distribution<double> randomReal(0.0, max_weight);
	uniform_int_distribution<int> randomInt(0, num_particles-1);
	int index = randomInt(gen);

	// get current weights
	vector<double> weights;
  	for (int i = 0; i < num_particles; i++) {
    	weights.push_back(particles[i].weight);
	}
	// get max weight
	max_weight = *max_element(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {

		beta += randomReal(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
