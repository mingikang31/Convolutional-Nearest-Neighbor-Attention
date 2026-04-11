library(tidyverse)
library(zoo)  # For rollapply
library(patchwork)

# Set working directory
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor-Attention/Final_csv/")

# Constants
SMOOTH_WINDOW <- 10

# ==========================================
# 0. DATA PREPARATION (Load & Process Once)
# ==========================================

# Read the main file
baseline_df <- read_csv("loss_test.csv")

# Process the data for both datasets at once
# This fixes the pivot_longer logic to match your new column structure
df_processed <- baseline_df %>%
  # 1. Filter for the specific layers you want
  filter(layer %in% c("Attention_NH1_s42", "ConvNNAttention_K9_N32_random_s42")) %>%
  
  # 2. Rename Layers for the Legend (Optional, but makes plot cleaner)
  mutate(Model_Label = case_when(
    layer == "Attention_NH1_s42" ~ "Attention",
    layer == "ConvNNAttention_K9_N32_random_s42" ~ "ConvNN",
    TRUE ~ layer
  )) %>%
  
  # 3. Pivot: Turn 'train_loss' and 'test_loss' columns into rows
  pivot_longer(
    cols = c(train_loss, test_loss),
    names_to = "Type_Raw",  # Temporary name
    values_to = "Loss"
  ) %>%
  
  # 4. Clean up Type names ("train_loss" -> "Train")
  mutate(Type = ifelse(Type_Raw == "train_loss", "Train", "Test")) %>%
  
  # 5. Apply Smoothing
  group_by(dataset, Model_Label, Type) %>%
  mutate(
    Loss_smooth = rollapply(Loss, width = SMOOTH_WINDOW, FUN = mean, align = "right", fill = NA, partial = TRUE)
  ) %>%
  ungroup()

# ==========================================
# PART 1: CIFAR-10 (Top Plot)
# ==========================================

plot_c10 <- df_processed %>%
  filter(dataset == "ViT-Tiny-CIFAR10") %>%
  ggplot(aes(x = epoch, y = Loss_smooth, color = Model_Label, linetype = Type)) +
  geom_line(linewidth = 1.0) +
  
  # Scales
  scale_linetype_manual(
    name = "Loss Type", 
    values = c("Train" = "dotted", "Test" = "solid")
  ) +
  # --- MANUAL COLOR SCALE FOR MODELS ---
  scale_color_manual(
    name = "Model",
    values = c(
      "Attention" = "#4DAF4A", # Set1 Blue
      "ConvNN"    = "#377EB8"  # Set1 Green
    )
  ) +  
  # Labels
  labs(
    title = "CIFAR-10", 
    x = NULL,           # Removed X Label for top plot
    y = "Loss"
  ) +
  
  # Theme
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    legend.title = element_text(face = "bold", size = 14), 
    legend.text = element_text(size = 14)
  )

# ==========================================
# PART 2: CIFAR-100 (Bottom Plot)
# ==========================================

plot_c100 <- df_processed %>%
  filter(dataset == "ViT-Tiny-CIFAR100") %>%
  ggplot(aes(x = epoch, y = Loss_smooth, color = Model_Label, linetype = Type)) +
  geom_line(linewidth = 1.0) +
  
  # Scales
  scale_linetype_manual(
    name = "Loss Type", 
    values = c("Train" = "dotted", "Test" = "solid")
  ) +
  # --- MANUAL COLOR SCALE FOR MODELS ---
  scale_color_manual(
    name = "Model",
    values = c(
      "Attention" = "#4DAF4A", # Set1 Blue
      "ConvNN"    = "#377EB8"  # Set1 Green
    )
  ) +
  # Labels
  labs(
    title = "CIFAR-100", 
    x = "Epochs", 
    y = "Loss"
  ) +
  
  # Theme
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    legend.title = element_text(face = "bold", size = 14), 
    legend.text = element_text(size = 14)
  )

# ==========================================
# PART 3: Combine and Save
# ==========================================

combined_plot = plot_c10 / plot_c100

final_plot = combined_plot + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom") 

print(final_plot)

# Save
ggsave(
  "/Users/mingikang/Developer/Convolutional-Nearest-Neighbor-Attention/Final_Plots/ViT_combined_loss_plot.png",
  plot = final_plot,
  width = 8,
  height = 5,
  units = "in",
  dpi = 500,
  bg = "white"
)