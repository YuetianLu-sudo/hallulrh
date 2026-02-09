# Core vs Periphery sanity check — product_by_company (test)

This report shows sampled triples split into `core_top` vs `periphery_bottom` using a TRAIN-defined threshold,
then evaluated on the chosen split (out-of-sample diagnostic).

## gemma_7b_it

### core_top (n=25)
Top companies: Microsoft(18), Apple(7)

| Δcos | subject (product) | gold_object (company) |
|---:|---|---|
| 0.752 | iPad Mini | Apple |
| 0.747 | iPod shuffle | Apple |
| 0.744 | iPod Touch | Apple |
| 0.741 | Newton OS | Apple |
| 0.739 | Visual Basic Script | Microsoft |
| 0.733 | Internet Explorer 11 | Microsoft |
| 0.733 | Vector Markup Language | Microsoft |
| 0.732 | NTFS | Microsoft |
| 0.729 | iOS 5 | Apple |
| 0.726 | Windows Communication Foundation | Microsoft |
| 0.723 | Windows Media Video | Microsoft |
| 0.722 | Windows Embedded CE 6.0 | Microsoft |

### periphery_bottom (n=25)
Top companies: Fiat(8), Sega(3), Square(2), Douglas(2), Adobe(2), Chrysler(2), Sears(1), Dodge(1)

| Δcos | subject (product) | gold_object (company) |
|---:|---|---|
| 0.338 | F3D Skyknight | Douglas |
| 0.349 | GAM-87 Skybolt | Douglas |
| 0.382 | Dardo IFV | Fiat |
| 0.405 | Final Fantasy V | Square |
| 0.436 | Final Fantasy Legend III | Square |
| 0.442 | Kenmore Appliances | Sears |
| 0.465 | M6 Gun Motor Carriage | Dodge |
| 0.471 | Alfa Romeo Brera | Fiat |
| 0.493 | Lancia Lybra | Fiat |
| 0.508 | Lancia Aprilia | Fiat |
| 0.514 | SEAT 133 | Fiat |
| 0.518 | Autobianchi Y10 | Fiat |

## llama3_1_8b_instruct

### core_top (n=25)
Top companies: Microsoft(17), Apple(7), IBM(1)

| Δcos | subject (product) | gold_object (company) |
|---:|---|---|
| 0.764 | NTFS | Microsoft |
| 0.764 | Channel Definition Format | Microsoft |
| 0.762 | Quick Look | Apple |
| 0.761 | QuickDraw | Apple |
| 0.761 | Visual Basic Script | Microsoft |
| 0.761 | High Performance File System | Microsoft |
| 0.757 | Newsstand | Apple |
| 0.757 | Uniform Type Identifier | Apple |
| 0.755 | VoiceOver | Apple |
| 0.754 | Virtual Hard Disk | Microsoft |
| 0.753 | Windows Embedded Compact | Microsoft |
| 0.749 | Windows Embedded | Microsoft |

### periphery_bottom (n=25)
Top companies: Fiat(8), Boeing(4), Sega(3), Chrysler(2), Square(2), Douglas(2), Dodge(1), Sears(1)

| Δcos | subject (product) | gold_object (company) |
|---:|---|---|
| 0.337 | Dardo IFV | Fiat |
| 0.487 | PGM-11 Redstone | Chrysler |
| 0.494 | Kenmore Appliances | Sears |
| 0.506 | PGM-19 Jupiter | Chrysler |
| 0.511 | Final Fantasy Legend III | Square |
| 0.512 | Final Fantasy V | Square |
| 0.516 | M6 Gun Motor Carriage | Dodge |
| 0.517 | Alfa Romeo Brera | Fiat |
| 0.524 | F3D Skyknight | Douglas |
| 0.528 | GAM-87 Skybolt | Douglas |
| 0.533 | Shining Hearts | Sega |
| 0.537 | Zelzal-2 | Iran |

## mistral_7b_instruct

### core_top (n=25)
Top companies: Apple(14), Microsoft(11)

| Δcos | subject (product) | gold_object (company) |
|---:|---|---|
| 0.779 | Uniform Type Identifier | Apple |
| 0.768 | Quick Look | Apple |
| 0.765 | Newsstand | Apple |
| 0.760 | Windows Embedded | Microsoft |
| 0.760 | Windows Embedded Compact | Microsoft |
| 0.754 | Windows Embedded Automotive | Microsoft |
| 0.747 | QuickDraw | Apple |
| 0.745 | MobileMe | Apple |
| 0.743 | VoiceOver | Apple |
| 0.740 | App Store | Apple |
| 0.740 | Logic Pro | Apple |
| 0.738 | CarPlay | Apple |

### periphery_bottom (n=25)
Top companies: Fiat(8), Boeing(3), Chrysler(2), Square(2), Sega(2), Sears(1), Dodge(1), Nokia(1)

| Δcos | subject (product) | gold_object (company) |
|---:|---|---|
| 0.082 | Dardo IFV | Fiat |
| 0.228 | PGM-19 Jupiter | Chrysler |
| 0.243 | PGM-11 Redstone | Chrysler |
| 0.290 | M6 Gun Motor Carriage | Dodge |
| 0.298 | Alfa Romeo Brera | Fiat |
| 0.330 | Kenmore Appliances | Sears |
| 0.331 | Shining Hearts | Sega |
| 0.350 | SEAT 133 | Fiat |
| 0.351 | Lancia Lybra | Fiat |
| 0.360 | Lancia Aprilia | Fiat |
| 0.361 | Lancia Dedra | Fiat |
| 0.362 | Lancia Flavia | Fiat |

## qwen2_5_7b_instruct

### core_top (n=25)
Top companies: Apple(10), Microsoft(9), Nintendo(3), Intel(2), Google(1)

| Δcos | subject (product) | gold_object (company) |
|---:|---|---|
| 0.606 | Windows Embedded Automotive | Microsoft |
| 0.601 | VoiceOver | Apple |
| 0.596 | Windows Embedded Compact | Microsoft |
| 0.592 | iPod | Apple |
| 0.592 | iPod | Apple |
| 0.590 | Windows Media Video | Microsoft |
| 0.590 | Xbox One | Microsoft |
| 0.589 | iPad Mini | Apple |
| 0.589 | Wii MotionPlus | Nintendo |
| 0.587 | NTFS | Microsoft |
| 0.587 | QuickDraw | Apple |
| 0.585 | Pentium III | Intel |

### periphery_bottom (n=25)
Top companies: Fiat(8), Sega(4), Square(2), Douglas(2), Chrysler(2), Dodge(1), Sony(1), Sears(1)

| Δcos | subject (product) | gold_object (company) |
|---:|---|---|
| 0.273 | Dardo IFV | Fiat |
| 0.325 | F3D Skyknight | Douglas |
| 0.344 | Final Fantasy Legend III | Square |
| 0.348 | M6 Gun Motor Carriage | Dodge |
| 0.352 | GAM-87 Skybolt | Douglas |
| 0.365 | Kenmore Appliances | Sears |
| 0.388 | Final Fantasy V | Square |
| 0.393 | CineAlta | Sony |
| 0.397 | Alfa Romeo Brera | Fiat |
| 0.398 | PGM-11 Redstone | Chrysler |
| 0.400 | Lancia Aprilia | Fiat |
| 0.408 | PGM-19 Jupiter | Chrysler |
