import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../models/analyze_response.dart';
import '../../screens/home/home_screen.dart';
import '../../screens/capture/capture_screen.dart';
import '../../screens/result/result_screen.dart';
import '../../screens/history/history_screen.dart';
import '../../screens/detail/swing_detail_screen.dart';
import '../../screens/settings/settings_screen.dart';

final _shellKey = GlobalKey<NavigatorState>();

final appRouter = GoRouter(
  initialLocation: '/',
  routes: [
    ShellRoute(
      navigatorKey: _shellKey,
      builder: (context, state, child) => _Shell(child: child),
      routes: [
        GoRoute(path: '/', builder: (_, __) => const HomeScreen()),
        GoRoute(path: '/history', builder: (_, __) => const HistoryScreen()),
        GoRoute(path: '/settings', builder: (_, __) => const SettingsScreen()),
      ],
    ),
    GoRoute(path: '/capture', builder: (_, __) => const CaptureScreen()),
    GoRoute(
      path: '/result',
      builder: (_, state) {
        final response = state.extra as AnalyzeResponse;
        return ResultScreen(response: response);
      },
    ),
    GoRoute(
      path: '/history/:id',
      builder: (_, state) =>
          SwingDetailScreen(swingId: state.pathParameters['id']!),
    ),
  ],
);

class _Shell extends StatefulWidget {
  final Widget child;
  const _Shell({required this.child});

  @override
  State<_Shell> createState() => _ShellState();
}

class _ShellState extends State<_Shell> {
  int _index = 0;

  static const _routes = ['/', '/history', '/settings'];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: widget.child,
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _index,
        onTap: (i) {
          setState(() => _index = i);
          context.go(_routes[i]);
        },
        items: const [
          BottomNavigationBarItem(
              icon: Icon(Icons.home_outlined),
              activeIcon: Icon(Icons.home),
              label: 'Home'),
          BottomNavigationBarItem(
              icon: Icon(Icons.history_outlined),
              activeIcon: Icon(Icons.history),
              label: 'History'),
          BottomNavigationBarItem(
              icon: Icon(Icons.settings_outlined),
              activeIcon: Icon(Icons.settings),
              label: 'Settings'),
        ],
      ),
    );
  }
}
