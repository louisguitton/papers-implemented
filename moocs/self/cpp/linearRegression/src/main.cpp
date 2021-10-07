#include "lsr.h"

int main()
{
    // X, y, print_debug messages
    simple_linear_regression slr({2, 3, 5, 7, 9}, {4, 5, 7, 10, 15}, true);
    slr.train();
    std::cout << slr.predict(8);
    slr.save_model("model.txt");
}
