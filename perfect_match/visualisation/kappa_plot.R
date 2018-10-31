# Copyright (C) 2018  Patrick Schwab, ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions
#  of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# This file plots the results of the News-8 treatment assignment bias experiment (Fig. 3 in the paper).

library(latex2exp)
makeTransparent = function(..., alpha=0.15) {
  # From: https://stackoverflow.com/a/20796068
  if(alpha<0 | alpha>1) stop("alpha must be between 0 and 1")

  alpha = floor(255*alpha)
  newColor = col2rgb(col=unlist(list(...)), alpha=FALSE)
  .makeTransparent = function(col, alpha) {
    rgb(red=col[1], green=col[2], blue=col[3], alpha=alpha, maxColorValue=255)
  }
  newColor = apply(newColor, 2, .makeTransparent, alpha=alpha)
  return(newColor)
}

# The following numbers were extracted from the finished experiments using run_results.sh.
pbm_mse_pehe <- c(18.50, 19.34, 20.83, 20.74, 22.28, 22.15, 22.56)
pbm_mse_pehe_std <- c(1.30, 1.47, 2.11, 1.53, 1.86, 2.02, 1.91)
pbm_mse_ate <- c(3.74, 4.82, 6.62, 6.69, 8.02, 8.48, 8.52)
pbm_mse_ate_std <- c(1.02, 1.10, 1.80, 1.53, 1.93, 2.20, 1.71)

pbm_pehe_pehe <- c(18.30, 19.40, 20.32, 21.90, 22.26, 22.15, 22.55)
pbm_pehe_pehe_std <- c(1.38, 1.63, 1.31, 1.85, 1.87, 1.85, 2.01)
pbm_pehe_ate <- c(3.85, 4.89, 5.98, 7.28, 8.00, 7.96, 8.94)
pbm_pehe_ate_std <- c(0.93, 1.22, 1.24, 1.45, 1.68, 2.11, 1.97)

tarnet_mse_pehe <- c(18.38, 19.61, 22.48, 22.71, 23.84, 24.12, 24.52)
tarnet_mse_pehe_std <- c(1.66, 1.27, 2.46, 1.66, 2.66, 2.18, 2.00)
tarnet_mse_ate <- c(4.74, 6.56, 9.43, 10.36, 11.29, 11.88, 12.40)
tarnet_mse_ate_std <- c(1.12, 0.96, 2.02, 1.53, 2.09, 1.77, 1.61)

cfrnet_mse_pehe <- c(18.23, 19.89, 21.71, 22.55, 23.60, 24.36, 24.02)
cfrnet_mse_pehe_std <- c(1.21, 2.12, 1.80, 2.29, 1.82, 1.90, 2.05)
cfrnet_mse_ate <- c(4.50, 6.69, 8.85, 10.34, 11.87, 12.40, 12.33)
cfrnet_mse_ate_std <- c(0.85, 1.59, 1.66, 1.60, 1.81, 1.93, 1.77)

tarnetpd_mse_pehe <- c(18.67, 19.75, 21.02, 21.56, 21.88, 23.34, 23.51)
tarnetpd_mse_pehe_std <- c(1.80, 1.78, 2.26, 2.49, 2.95, 3.28, 2.88)
tarnetpd_mse_ate <- c(5.04, 5.96, 6.99, 7.56, 7.87, 9.64, 9.48)
tarnetpd_mse_ate_std <- c(1.65, 2.19, 2.75, 2.94, 3.30, 4.54, 3.78)

cf_mse_pehe <- c(22.62, 23.92, 25.76, 26.79, 27.31, 27.48, 27.54)
cf_mse_pehe_std <- c(1.67, 1.77, 2.05, 2.43, 2.53, 2.49, 1.98)
cf_mse_ate <- c(12.25, 13.51, 15.12, 16.13, 16.52, 16.82, 16.76)
cf_mse_ate_std <- c(1.56, 1.65, 2.20, 2.43, 2.78, 2.47, 2.25)

kappas <- c(5, 7, 10, 12, 15, 17, 20)
pehe_min <- 16.3
pehe_max <- 30
pehe_title <- TeX("$\\sqrt{\\epsilon_{PEHE}}$")
pehe_title_cex <- 3

ate_min <- 2.5
ate_max <- 20
ate_title <- TeX("$\\epsilon_{ATE}$")
ate_title_cex <- 3

for(metric in c("pehe", "ate")) {
  # Plot setup.
  pdf(paste("kappa_", metric, ".pdf", sep=""), width=14)
  par(mar = c(5.1, 5.6, 4.1, 2.1))

  # Get per-metric config.
  min_y <- get(paste(metric, "_min", sep=""))
  max_y <- get(paste(metric, "_max", sep=""))
  title <- get(paste(metric, "_title", sep=""))
  title_cex <- get(paste(metric, "_title_cex", sep=""))

  plot(kappas, cf_mse_pehe, type = 'n', xlim=c(min(kappas), max(kappas)), ylim=c(min_y, max_y),
       cex.axis=2, ann=FALSE, mgp=c(3, 1, 0), xaxs = "i", yaxs = "i")
  mtext(side=1, text=substitute(paste("Treatment Assignment Bias ", kappa)), line=4, cex=3)
  mtext(side=2, text=title, line=2.75, cex=title_cex)

  # Plot each method.
  i <- 1
  postfix_mean <- paste("_", metric, sep="")
  postfix_std <- paste("_", metric, "_std", sep="")
  colors <- c('#F59799', '#9DC7EA', '#FDC67C', '#A75CC6', '#A7916D', '#666666')
  methods <- c("pbm_mse", "cfrnet_mse", "tarnet_mse", "tarnetpd_mse", "cf_mse")
  names <- c("PM", "CFRNET", "TARNET", "PD", "CF")
  for(method in methods) {
    mean <- get(paste(method, postfix_mean, sep=""))
    std <- get(paste(method, postfix_std, sep=""))
    high <- mean + std
    low <- mean - std

    polygon(c(kappas, rev(kappas)), c(high, rev(low)),
            col=makeTransparent(colors[i]), border = NA)

    lines(kappas, mean, lwd=5, cex=3, pch=i-1, type="o", col=colors[i])
    lines(kappas, high, lwd=1, lty="dashed", col=colors[i])
    lines(kappas, low, lwd=1, lty="dashed", col=colors[i])

    i <- i + 1
  }
  legend("topleft", legend=names, pch=0:length(methods), col=colors, cex=2, lwd=5)
  dev.off()
}
