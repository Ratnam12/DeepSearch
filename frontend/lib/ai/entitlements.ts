// Clerk owns identity, so there is no per-user-type tier — every signed-in
// user gets the default entitlements. If we add tiers later (e.g. via Clerk
// public metadata for "pro" subscribers), branch on that here.

type Entitlements = {
  maxMessagesPerHour: number;
};

export const defaultEntitlements: Entitlements = {
  maxMessagesPerHour: 50,
};
