#pragma once
#ifndef CSAPS_H
#define CSAPS_H

#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

/* example: https://github.com/sergibro/SmoothingCubicSpline/blob/master/CppWrapper/smooth_cubic_spline_utils.hpp

// init the spline
const size_t pcount = 9;
csaps::DoubleArray xdata(pcount);
xdata << 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
csaps::DoubleArray ydata(pcount);
ydata << 0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0;
csaps::UnivariateCubicSmoothingSpline sp(xdata, ydata, 1.0);

// array input
const size_t xidata_size = 8;
csaps::DoubleArray xidata(xidata_size);
xidata << 1, 2, 3;
csaps::DoubleArray yidata = sp.Evaluate(xidata, 1);
size_t yi_size = yidata.size();
for (size_t i = 0; i < yi_size; ++i) {
std::cout << " " << yidata(i);
}

// singal input
double x_test_val = 7;
double y_test_val = sp.Evaluate(x_test_val, 1);
std::cout << "\nsingle test: " << y_test_val;

*/


namespace csaps
{
using Index = Eigen::DenseIndex;
using Size = Index;
using DoubleArray = Eigen::ArrayXd;
using DoubleArray2D = Eigen::ArrayXXd;
using IndexArray = Eigen::Array<Index, Eigen::Dynamic, 1>;
using DoubleSparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, Index>;

using DoubleLimits = std::numeric_limits<double>;


//! Calculates the 1-th discrete difference
DoubleArray Diff(const DoubleArray &vec);


//! Returns the indices of the bins to which each value in input array belongs
IndexArray Digitize(const DoubleArray &arr, const DoubleArray &bins);


//! Makes rows x cols sparse matrix from diagonals and offsets
DoubleSparseMatrix MakeSparseDiagMatrix(const DoubleArray2D& diags, const IndexArray& offsets, Size rows, Size cols);


//! Solves sparse linear system Ab = x via supernodal LU factorization
const DoubleArray SolveLinearSystem(const DoubleSparseMatrix &A, DoubleArray &b);


class UnivariateCubicSmoothingSpline
{
public:
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, double smooth);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights, double smooth);

  DoubleArray operator()(const DoubleArray &xidata);
  DoubleArray operator()(const Size pcount, DoubleArray &xidata);

  double GetSmooth() const { return m_smooth; }
  const DoubleArray& GetBreaks() const { return m_xdata; }
  const DoubleArray2D& GetCoeffs() const { return m_coeffs; }
  Index GetPieces() const { return m_coeffs.rows(); }
  double Evaluate(const double &x_val, int deriv = 0);
  DoubleArray Evaluate(const DoubleArray &xidata, int deriv = 0);

protected:
  void MakeSpline();
  void InitEdge();

protected:
  DoubleArray m_xdata;
  DoubleArray m_ydata;
  DoubleArray m_weights;
  DoubleArray m_edges;

  double m_smooth;

  DoubleArray2D m_coeffs;
};

} // namespace csaps

#endif // CSAPS_H
