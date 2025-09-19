#' Plot Cross-Validation Results
#' @return A ggplot object.
#' @export
plot.cbSCRIP.cv <- function(x, ...) {
    
    mean_n_vars <- rowMeans(x$non_zero_matrix, na.rm = TRUE)
    
    plot_data <- data.frame(
        lambda = x$lambdagrid,
        mean_dev = x$deviance_mean,
        upper = x$deviance_mean + x$deviance_se,
        lower = x$deviance_mean - x$deviance_se,
        n_vars = round(mean_n_vars)
    )
    
    n_total <- nrow(plot_data)
    n_labels <- min(n_total, 10) # Show at most 10 labels
    # Select ~10 evenly spaced indices from the data
    label_indices <- round(seq(1, n_total, length.out = n_labels))
    axis_labels_data <- plot_data[label_indices, ]
    
    ggplot(plot_data, aes(x = lambda, y = mean_dev)) +
        geom_errorbar(aes(ymin = lower, ymax = upper), 
                      width = 0.05, 
                      color = "grey80") +
        geom_point(color = "#f94144",
                   alpha = 1) +
        geom_vline(aes(xintercept = x$lambda.min,
                       color = "Lambda.min",
                       linetype = "Lambda.min")) +
        geom_vline(aes(xintercept = x$lambda.1se,
                       color = "Lambda.1se",
                       linetype = "Lambda.1se")) +
        scale_color_manual(
            name = NULL,
            values = c("Lambda.min" = "#277DA1", "Lambda.1se" = "#264653")
        ) +
        scale_linetype_manual(
            name = NULL,
            values = c("Lambda.min" = "dashed", "Lambda.1se" = "dotted")
        ) +
        labs(
            x = "Lambda",
            y = "Multinomial Deviance",
            title = "Cross-Validation Performance",
            color = "",
            linetype = ""
        ) +
        scale_x_log10(
            sec.axis = sec_axis(
                trans = ~.,
                name = "Mean Number of Selected Variables",
                breaks = axis_labels_data$lambda,  # Position ticks at these lambda values
                labels = axis_labels_data$n_vars   # Use n_vars as the label text
            )
        ) +
        theme_minimal()
}


#' Plot Coefficient Paths from a cb.path Object
#'
#' S3 method to plot the regularization path of coefficients.
#'
#' @param x An object of class `cb.path`.
#' @param plot_intercept Logical. Whether to include the intercept in the plot.
#' @param ... Not used.
#'
#' @return A ggplot object.
#' @export
plot.cbSCRIP <- function(x, plot_intercept = FALSE, ...) {
    
    # Wrangle the list of coefficient matrices into a long-format tibble
    plot_data <- imap_dfr(x$coefficients, ~{
        .x %>%
            as.data.frame() %>%
            tibble::rownames_to_column("variable") %>%
            mutate(lambda = as.numeric(.y))
    }) %>%
        pivot_longer(
            cols = -c(variable, lambda),
            names_to = "event_type",
            values_to = "coefficient"
        )
    
    if (!plot_intercept) {
        plot_data <- filter(plot_data, variable != "(Intercept)")
    }
    
    ggplot(plot_data, aes(x = lambda, y = coefficient, group = variable, color = variable)) +
        geom_line(alpha = 0.8) +
        facet_wrap(~event_type, scales = "free_y") +
        theme_minimal() +
        guides(color = "none") + # Hide legend for clarity if many variables
        labs(
            x = "Lambda",
            y = "Coefficient Value",
            title = "Coefficient Regularization Paths",
            subtitle = "Each line represents a variable's coefficient as penalty increases"
        ) +
        scale_x_log10()
}