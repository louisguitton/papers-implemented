#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>


class simple_linear_regression
{
private:
    unsigned int N = 0.0;

    std::vector<double> X;
    std::vector<double> y;

    std::vector<double> XX;
    std::vector<double> Xy;

    bool verbose;

    double sigma_X = 0.0;
    double sigma_y = 0.0;
    double sigma_XX = 0.0;
    double sigma_Xy = 0.0;

    double m = 0.0;

    double b = 0.0;

    void print(std::string message);

    void calculate_N();

    void x_square();
    void x_cross_y();

    void calculate_sigma();

    void calculate_slope();

    void calculate_bias();

public:
    simple_linear_regression(std::string model_name);
    simple_linear_regression(std::vector<double> X, std::vector<double> y, bool verbose = false);
    void train();
    double predict(double _X);
    void save_model(std::string file_name);
};
