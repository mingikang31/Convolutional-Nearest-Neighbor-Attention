library(tidyverse)
library(stringr)
library(RColorBrewer) # Needed for brewer.pal()

# --- Data Reading ---
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor-Attention/Final_Output/")
df <- read_csv("csv/N_test.csv")

df_cifar = df %>% 
  filter(dataset == "ViT-Tiny-CIFAR10") %>%
  select(layer, 
         type,
         N, 
         BestAcc = best_test_accuracy_top1, 
         AveAvv = ave_test_accuracy_top1
   ) 

attention = df_cifar %>%
  filter(layer == "Attention")

convnn_all = df_cifar %>%
  filter(layer == "ConvNNAttention" & type == "all")





# print(n_plot)
# 
# save_path <- "/Users/mingikang/Developer/Convolutional-Nearest-Neighbor-Attention/Final_Output/plots/cifar10_n_test.png"
# 
# # Step 3: (Optional) Save the plot
# ggsave(
#   save_path,
#   plot = n_plot,
#   width = 6,
#   height = 6,
#   units = "in",
#   dpi = 300,
#   bg = "white"
# )