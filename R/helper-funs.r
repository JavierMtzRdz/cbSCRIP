#' @export
same <- function (x, y, tolerance = .Machine$double.eps) {
    abs(x - y) < tolerance
}