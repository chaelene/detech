import { test, expect } from '@playwright/test';

test.describe('DETECH Stream', () => {
  test('should load the main page', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('h1')).toContainText('DETECH');
  });

  test('should display stream controls', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByTestId('stream-toggle')).toHaveText(/Start Stream/i);
  });

  test('should handle wallet connection', async ({ page }) => {
    await page.goto('/');
    // TODO: Implement wallet connection test
    // This will require mocking WalletConnect
  });

  test('surface detection alerts when mock payload dispatched', async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => {
      const detail = {
        session_id: 'test-session',
        alert_id: 'alert-123',
        timestamp: Date.now(),
        severity: 'high',
        type: 'gesture-bridge',
        description: 'Mock person detection',
        commands: ['swarm-follow'],
        detection: {
          timestamp: Date.now(),
          objects: [
            {
              name: 'person',
              confidence: 0.94,
              bbox: { x: 0.2, y: 0.2, width: 0.3, height: 0.4 },
            },
          ],
          gestures: [
            {
              name: 'raise-hand',
              confidence: 0.88,
            },
          ],
        },
        payment_status: 'confirmed',
      } satisfies Record<string, unknown>;

      window.dispatchEvent(new CustomEvent('detech:mock-alert', { detail }));
    });

    await expect(page.getByTestId('swarm-alert-toast')).toBeVisible();
    await expect(page.getByTestId('alerts-overlay')).toBeVisible();
  });
});
