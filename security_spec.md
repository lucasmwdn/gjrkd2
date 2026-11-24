# Security Specification & Payload-First TDD (Phase 0)

## 1. Data Invariants
1. **Criteria Integrity**: `/criteria/active` is writable ONLY by verified admin (`luc.asm.wdn@gmail.com`) and MUST strictly lock the four area max scores (`구체성=30`, `균형성=20`, `과장도=25`, `일관성=25`) and `baseTrustScore` (`0..100`).
2. **Immutable Audit Trail**: `/criteriaHistory/{historyId}` can only be created by verified admins with server timestamp `request.time` and cannot be updated or deleted.
3. **Appeal Ownership & State Transition**: `/appeals/{appealId}` can only be created by email-verified users with `authorId == request.auth.uid`, `createdAt == request.time`, and initial `status == '대기'`. Only admins can transition status to `'반영됨'` or `'검토 완료'` or attach `reanalysisComparison`.
4. **Proposal Threshold & Guardrails**: `/proposals/{proposalId}` can only be created/updated by verified admins, MUST have `supportingAppealCount >= 3`, and new signals (`add_signal`) MUST have `proposedPoints` strictly within `[-10, -3]` or `[3, 10]`. Once `status` leaves `'pending'`, terminal state locking prevents re-transition.
5. **Zero PII Exposure**: No reviewer nicknames, emails, phone numbers, or addresses are stored in `appeals`, `criteria`, `criteriaHistory`, or `proposals`.

## 2. The "Dirty Dozen" Adversarial Payloads (Must All Return `PERMISSION_DENIED`)
1. **Unauthenticated Criteria Write**: Unauthenticated client attempts `setDoc('/criteria/active', {...})`. -> Denied (`isAdmin()` false).
2. **Non-Admin Criteria Tampering**: Regular signed-in user attempts `updateDoc('/criteria/active', { version: 99 })`. -> Denied (`isAdmin()` false).
3. **Spoofed Admin Unverified Email**: User with `email: 'luc.asm.wdn@gmail.com'` and `email_verified: false` attempts admin write. -> Denied (`isVerifiedUser()` false).
4. **Area Cap Bypass**: Admin attempts to raise `구체성` `maxScore` from `30` to `50` in `/criteria/active`. -> Denied (`isValidCriteriaConfig()` enforces exact caps).
5. **Shadow Field Injection on Appeal**: Verified user creates `/appeals/a1` with extra ghost field `{ isVerifiedAdmin: true }`. -> Denied (`keys().hasOnly(...)`).
6. **Appeal Author Spoofing**: Verified user creates `/appeals/a2` with `authorId: 'someone-else'`. -> Denied (`incoming().authorId == request.auth.uid`).
7. **Appeal Pre-Approved State Shortcutting**: Verified user creates `/appeals/a3` with `status: '반영됨'`. -> Denied (`incoming().status == '대기'`).
8. **Client Timestamp Forgery**: Verified user creates `/appeals/a4` with backdated `createdAt` (`!= request.time`). -> Denied (`incoming().createdAt == request.time`).
9. **ID Poisoning Attack**: User attempts `setDoc('/appeals/' + 'x'.repeat(200), {...})`. -> Denied (`isValidId()` length & regex guard).
10. **Non-Admin Proposal Creation**: Regular user attempts `setDoc('/proposals/p1', {...})`. -> Denied (`isAdmin()` false).
11. **Out-of-Bounds New Signal Score**: Admin creates proposal with `proposedAction: 'add_signal'` and `proposedPoints: 25` (violating `±3~±10`). -> Denied (`isValidProposal()`).
12. **Low-Support Proposal Bypass**: Admin creates proposal with `supportingAppealCount: 1` (violating `>= 3`). -> Denied (`isValidProposal()`).
