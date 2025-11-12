library(tidyverse)
library(stringr)
library(RColorBrewer) # Needed for brewer.pal()

### K: 1-12
# --- Data Reading ---
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor-Attention/Final_Output/")
df <- read_csv("csv/K_test.csv")

# --- Data Preparation ---
df_cifar = df %>%
  filter(dataset == "ViT-Tiny-CIFAR100") %>%
  select(layer,
         K,
         BestAcc = best_test_accuracy_top1,
         AveAcc = ave_test_accuracy_top1
  )

# Split data
attention = df_cifar %>%
  filter(layer == "Attention")


convnn = df_cifar %>%
  filter(layer != "Attention")  %>%
  mutate(K = as.numeric(K)) %>% # *** FIX 1: Convert K to a number ***
  filter(K != 16 & K != 25 & K != 36) # Filter using numbers

# --- Setup for Scales (to merge legends) ---

# Get all unique layer names
convnn_layers = unique(convnn$layer)
all_layers = c(convnn_layers, "Attention")

# 1. Define linetypes: "Attention" is dotted, others solid
linetype_values = setNames(
  c(rep("solid", length(convnn_layers)), "solid"),
  all_layers
)
# 2. Define colors (REVISED)
n_convnn_layers = length(convnn_layers)

# Get a palette of at least 3 colors, or more if needed
base_palette = brewer.pal(max(3, n_convnn_layers), "Set1")

# Take only the first n colors from that palette
convnn_colors = base_palette[1:n_convnn_layers]

# Assign colors: "Attention" is black, others from the palette
color_values = setNames(
  c(convnn_colors, "black"),
  all_layers
)

# --- Create the Plot ---
k_plot <- ggplot() + # Start with a blank canvas

  # Layer 1: ConvNN lines
  geom_line(
    data = convnn,
    aes(x = K,
        y = AveAcc,
        color = factor(layer),
        linetype = factor(layer)
    ),
    linewidth = 1
  ) +

  # Layer 2: Attention horizontal line
  geom_hline(
    data = attention,
    aes(yintercept = AveAcc,
        color = factor(layer),
        linetype = factor(layer)   # Use factor() for consistency
    ),
    linewidth = 1.0
  ) +

  # --- Customize Scales and Legends ---

  # Use scale_color_manual to control all colors
  scale_color_manual(
    name = "Layer Type", # Renamed legend title
    values = color_values
  ) +

  # Use scale_linetype_manual to control all linetypes
  scale_linetype_manual(
    name = "Layer Type", # Use same name to merge legends
    values = linetype_values
  ) +

  # --- Customize Axes and Labels ---
  scale_x_continuous(
    breaks = seq(1, 12, by = 1),
    minor_breaks = NULL,
    limits = c(1, 12)
  ) +
  scale_y_continuous(
    # breaks = seq(46, 51, by=0.5),
    # minor_breaks = NULL
  ) +
  labs(
    x = "K (Number of Neighbors)",
    y = "Top-1 Accuracy (%)"
  ) +

  # --- Apply a Theme ---
  theme_bw(base_size = 12) +
  theme(
    legend.position = "right",
    legend.title = element_text(face = "bold")
  )


# 
# # --- Data Preparation ---
# df_cifar = df %>% 
#   filter(dataset == "ViT-Tiny-CIFAR10") %>%
#   select(layer, 
#          K, 
#          BestAcc = best_test_accuracy_top1, 
#          AveAcc = ave_test_accuracy_top1
#   ) 
# 
# # Split data
# attention = df_cifar %>%
#   filter(layer == "Attention") 
# 
# 
# convnn = df_cifar %>%
#   filter(layer != "Attention")  %>% 
#   mutate(K = as.numeric(K)) %>%
#   filter(K %in% c(1, 4, 9, 16, 25, 36))
# 
# # --- Setup for Scales (to merge legends) ---
# 
# # Get all unique layer names
# convnn_layers = unique(convnn$layer)
# all_layers = c(convnn_layers, "Attention")
# 
# # 1. Define linetypes: "Attention" is dotted, others solid
# linetype_values = setNames(
#   c(rep("solid", length(convnn_layers)), "solid"), 
#   all_layers
# )
# # 2. Define colors (REVISED)
# n_convnn_layers = length(convnn_layers)
# 
# # Get a palette of at least 3 colors, or more if needed
# base_palette = brewer.pal(max(3, n_convnn_layers), "Set1")
# 
# # Take only the first n colors from that palette
# convnn_colors = base_palette[1:n_convnn_layers]
# 
# # Assign colors: "Attention" is black, others from the palette
# color_values = setNames(
#   c(convnn_colors, "black"),
#   all_layers
# )
# 
# # --- Create the Plot ---
# k_plot <- ggplot() + # Start with a blank canvas
#   
#   # Layer 1: ConvNN lines
#   geom_line(
#     data = convnn, 
#     aes(x = K, 
#         y = AveAcc, 
#         color = factor(layer), 
#         linetype = factor(layer)
#     ), 
#     linewidth = 1
#   ) +
#   
#   # Layer 2: Attention horizontal line
#   geom_hline(
#     data = attention, 
#     aes(yintercept = AveAcc, 
#         color = factor(layer),     
#         linetype = factor(layer)   # Use factor() for consistency
#     ), 
#     linewidth = 1.0
#   ) +
#   
#   # --- Customize Scales and Legends ---
#   
#   # Use scale_color_manual to control all colors
#   scale_color_manual(
#     name = "Layer Type", # Renamed legend title
#     values = color_values
#   ) +
#   
#   # Use scale_linetype_manual to control all linetypes
#   scale_linetype_manual(
#     name = "Layer Type", # Use same name to merge legends
#     values = linetype_values
#   ) +
#   
#   # --- Customize Axes and Labels ---
#   scale_x_continuous(
#     breaks = c(1, 4, 9, 16, 25, 36),
#     minor_breaks = NULL, 
#     limits = c(1, 36)
#   ) + 
#   scale_y_continuous(
#     breaks = seq(73, 79, by=0.5),
#     minor_breaks = NULL
#   ) +
#   # scale_y_continuous(
#   #    breaks = seq(46, 51, by=0.5), 
#   #    minor_breaks = NULL
#   # ) + 
#   labs(
#     x = "K (Number of Neighbors)",
#     y = "Top-1 Accuracy (%)"
#   ) +
#   
#   # --- Apply a Theme ---
#   theme_bw(base_size = 12) +
#   theme(
#     legend.position = "right",
#     legend.title = element_text(face = "bold")
#   )

# Print the plot
print(k_plot)


save_path <- "/Users/mingikang/Developer/Convolutional-Nearest-Neighbor-Attention/Final_Output/plots/cifar100_k_test.png"

# Step 3: (Optional) Save the plot
ggsave(
  save_path,
  plot = k_plot,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300,
  bg = "white"
)
