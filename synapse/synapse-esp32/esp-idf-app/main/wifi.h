#ifndef WIFI_H
#define WIFI_H

#include "esp_err.h"

/**
 * Initialize WiFi in STA mode and connect to the given AP.
 * On ESP32-P4 this routes through the companion ESP32-C6 over SDIO.
 *
 * Blocks until connected (up to 15 s timeout).
 * Returns ESP_OK on success, ESP_FAIL on timeout or max retries.
 */
esp_err_t wifi_init_sta(const char *ssid, const char *password);

/**
 * Return the IPv4 address string assigned by DHCP after a successful
 * wifi_init_sta() call.  Returns "" if not yet connected.
 */
const char *wifi_get_ip(void);

#endif /* WIFI_H */
