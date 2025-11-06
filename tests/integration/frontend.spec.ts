import { expect, test } from '@playwright/test';

const FRONTEND_ORIGIN = process.env.FRONTEND_URL ?? 'http://localhost:3000';

test.describe('DETECH live console', () => {
  test('renders swarm-evolved alert with payment details', async ({ page }) => {
    await page.goto(FRONTEND_ORIGIN, { waitUntil: 'networkidle' });
    await expect(page.getByText('DETECH Live Ops Console')).toBeVisible();
    await expect(page.getByText('Swarm Alerts')).toBeVisible();

    const swarmPayload = {
      session_id: 'pw-session',
      severity: 'medium',
      description: 'Swarm elevated perimeter breach',
      timestamp: Date.now(),
      type: 'edge_detection',
      swarm_confidence: 0.84,
      commands: ['dispatch-drone'],
      metrics: { ingest_latency_ms: 420.2 },
      detection: {
        timestamp: Date.now(),
        objects: [
          {
            name: 'person',
            confidence: 0.9,
            bbox: [120, 160, 180, 240],
          },
        ],
        gestures: [],
      },
      swarm: {
        refined: {
          accuracy: 0.84,
          confidence: 0.84,
          anomaly_score: 0.78,
          threat_level: 'elevated',
        },
        commands: ['dispatch-drone'],
        evolution: {
          baseline_confidence: 0.62,
          refined_accuracy: 0.84,
          delta: 0.22,
        },
      },
      x402: {
        status: 'charged',
        charged_amount: '0.050000',
        tx_signature: 'MOCK-PLAYWRIGHT',
      },
      payment_status: 'confirmed',
      payment_amount: '0.050000',
    };

    await page.evaluate((detail) => {
      window.dispatchEvent(new CustomEvent('detech:mock-alert', { detail }));
    }, swarmPayload);

    const alertCard = page.getByTestId('alert-card').first();
    await expect(alertCard).toContainText('dispatch-drone');
    await expect(alertCard).toContainText('Charged 0.05 USDC');

    const overlay = page.getByTestId('alerts-overlay');
    await expect(overlay).toContainText('person • 90%');

    const toast = page.getByTestId('swarm-alert-toast');
    await expect(toast).toContainText('Swarm Alert');
    await expect(toast).toContainText('dispatch-drone');
  });
});

