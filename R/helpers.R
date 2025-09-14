same <- function (x, y, tolerance = .Machine$double.eps^0.5) {
    abs(x - y) < tolerance
}