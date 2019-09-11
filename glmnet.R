library(data.table)
library(glmnet)


message("Read train")
tr <- fread("train.csv")
message("Read test")
te <- fread("test.csv")


message("Munge")
y <- tr$target
tr <- tr[, 1:24]
tre <- rbind(tr, te)
tre[, month := as.character(month)]
tre[, day := as.character(day)]
tre[, ord_0 := as.character(ord_0)]
tre[, bin_2 := as.character(bin_2)]
tre[, bin_1 := as.character(bin_1)]
tre[, bin_0 := as.character(bin_0)]


message("Make matrix")
tre_0 <- sparse.model.matrix(~ ., data = tre[, 2:24])
tre_0 <- tre_0[, 2:16530]
pf_0 <- colSums(tre_0)
tre_0 <- tre_0[, pf_0 > 5]

tr_0 <- tre_0[1:nrow(tr), ]
te_0 <- tre_0[(nrow(tr) + 1):(nrow(tr) + nrow(te)), ]

pf <- colSums(tr_0)
set.seed(1234)
message("CV GLMNET")
m_cv <- cv.glmnet(tr_0, y, family = "binomial", type.measure = "auc", alpha = 0, 
                  standardize = FALSE, lambda = seq(0.000035, 0.000030, -0.0000002),
                  penalty.factor = (1 / pf) ^ 0.1, thresh = 1e-10, maxit = 1e9,
									keep = TRUE, nfolds = 100)

print(m_cv$cvm)
browser()

pr <- predict(m_cv$glmnet.fit, te_0, s = m_cv$lambda.min * 0.9, type = "response")
colnames(pr) <- "target"
submission <- data.frame(id = te$id, target = pr, row.names = NULL)
fwrite(submission, file="submission_glmnet.csv")
