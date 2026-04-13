import 'package:flutter/material.dart';

class AppTheme {
  static const Color good = Color(0xFF4CAF50);
  static const Color bad = Color(0xFFF44336);
  static const Color accent = Color(0xFF2196F3);
  static const Color background = Color(0xFF121212);
  static const Color surface = Color(0xFF1E1E1E);
  static const Color onSurface = Color(0xFFE0E0E0);

  static ThemeData get dark => ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: background,
        colorScheme: const ColorScheme.dark(
          primary: accent,
          secondary: good,
          error: bad,
          surface: surface,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: surface,
          elevation: 0,
          titleTextStyle: TextStyle(
            color: onSurface,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
        ),
        cardTheme: CardThemeData(
          color: surface,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          elevation: 2,
        ),
        bottomNavigationBarTheme: const BottomNavigationBarThemeData(
          backgroundColor: surface,
          selectedItemColor: accent,
          unselectedItemColor: Colors.grey,
        ),
      );
}
