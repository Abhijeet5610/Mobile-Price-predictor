# Predict Mobile Phone Pricing

A machine learning project to predict the price range of mobile phones (low, medium, high, very high) based on their technical specifications. This repository includes data exploration, multiple model comparisons (including custom implementations), and deployment-ready code.

---

## Objective

Build a system that can predict the price category of a mobile phone using its features. The goal is to classify each phone into one of four price ranges:  
- 0 = Low cost  
- 1 = Medium cost  
- 2 = High cost  
- 3 = Very high cost

---

## Dataset

The dataset contains various features of mobile phones currently available in the market.  
**Target Column:** `price_range` (0, 1, 2, 3)

**Feature Descriptions:**
- `battery_power`: Battery capacity in mAh
- `blue`: Has Bluetooth (1: Yes, 0: No)
- `clock_speed`: Processor speed (in GHz)
- `dual_sim`: Dual SIM support (1: Yes, 0: No)
- `fc`: Front camera megapixels
- `four_g`: Has 4G (1: Yes, 0: No)
- `int_memory`: Internal memory in GB
- `m_deep`: Mobile depth in cm
- `mobile_wt`: Weight in grams
- `n_cores`: Number of processor cores
- `pc`: Primary camera megapixels
- `px_height`: Pixel resolution height
- `px_width`: Pixel resolution width
- `ram`: RAM in MB
- `sc_h`: Screen height in cm
- `sc_w`: Screen width in cm
- `talk_time`: Battery talk time in hours
- `three_g`: Has 3G (1: Yes, 0: No)
- `touch_screen`: Has touch screen (1: Yes, 0: No)
- `wifi`: Has WiFi (1: Yes, 0: No)

---


