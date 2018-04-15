wendys = read.csv("~/Documents/TO412/finalproject/wendys1.csv")
arbys = read.csv("~/Documents/TO412/finalproject/arbys1.csv")
mcD = read.csv("~/Documents/TO412/finalproject/mcdonalds1.csv")
bk = read.csv("~/Documents/TO412/finalproject/burgerking1.csv")
names(wendys)[24] = "google"
names(arbys)[24] = "google"
names(mcD)[24] = "google"
names(bk)[24] = "google"
wendys = rbind(wendys, arbys, mcD, bk)
wendys = wendys[,c(-1,-2,-3)]
wendys$tot = (wendys$re * 3) + wendys$fav
wendys = wendys[,c(-1, -14)]
linearModel = lm(tot ~ created + followC + sn + pic + polarity + subjectivity + analyzer + atCompany + inReply + ats + pounds + links + sent + google, data = wendys )
linearModel2 = step(linearModel, direction = "backward")
summary(linearModel2)
coefs = as.data.frame(coef(linearModel2))
coefs$variable = rownames(coefs)
names(coefs)[1] = "coefficients"
coefs = cbind(coefs, summary(linearModel2)$coefficients[,4])
names(coefs)[3] = "P-Value"
rownames(coefs) = NULL
write.csv(coefs, "~/Documents/TO412/finalproject/coefficients.csv")

