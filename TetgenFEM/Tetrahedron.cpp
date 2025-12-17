#include "Tetrahedron.h"

#include <algorithm>
#include <limits>

namespace {
float clampNuForStableCompliance(float E1, float E2, float E3, float nu) {
	E1 = std::max(E1, 1e-12f);
	E2 = std::max(E2, 1e-12f);
	E3 = std::max(E3, 1e-12f);
	nu = std::min(0.49f, std::max(0.0f, nu));

	auto minEigenNormalBlock = [&](float candidate) -> float {
		Eigen::Matrix3f Sn = Eigen::Matrix3f::Zero();
		Sn(0, 0) = 1.0f / E1;
		Sn(1, 1) = 1.0f / E2;
		Sn(2, 2) = 1.0f / E3;
		Sn(0, 1) = Sn(1, 0) = -candidate / E1;
		Sn(0, 2) = Sn(2, 0) = -candidate / E1;
		Sn(1, 2) = Sn(2, 1) = -candidate / E2;
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(Sn);
		if (es.info() != Eigen::Success) return -1.0f;
		return es.eigenvalues().minCoeff();
	};

	const float target = 1e-10f;
	if (minEigenNormalBlock(nu) > target) return nu;

	float lo = 0.0f;
	float hi = nu;
	for (int iter = 0; iter < 30; ++iter) {
		const float mid = 0.5f * (lo + hi);
		if (minEigenNormalBlock(mid) > target) lo = mid;
		else hi = mid;
	}
	return lo;
}

Eigen::Matrix<float, 6, 6> projectSPD(const Eigen::Matrix<float, 6, 6>& M) {
	Eigen::Matrix<float, 6, 6> sym = 0.5f * (M + M.transpose());
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> es(sym);
	if (es.info() != Eigen::Success) {
		Eigen::Matrix<float, 6, 6> reg = sym;
		reg.diagonal().array() += 1e-6f;
		return reg;
	}
	Eigen::Matrix<float, 6, 1> evals = es.eigenvalues();
	const float maxEval = std::max(1.0f, evals.maxCoeff());
	const float minEval = std::max(1e-6f, maxEval * 1e-7f);
	for (int i = 0; i < evals.size(); ++i) {
		if (!(evals[i] >= minEval)) evals[i] = minEval;
	}
	return es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();
}
} // namespace

Eigen::MatrixXf Tetrahedron::createElementKAni(float E1, float E2, float E3, float nu, const Eigen::Vector3f& groupCenterOfMass)
{
	float x1 = vertices[0]->x - groupCenterOfMass.x();
	float y1 = vertices[0]->y - groupCenterOfMass.y();
	float z1 = vertices[0]->z - groupCenterOfMass.z();
	float x2 = vertices[1]->x - groupCenterOfMass.x();
	float y2 = vertices[1]->y - groupCenterOfMass.y();
	float z2 = vertices[1]->z - groupCenterOfMass.z();
	float x3 = vertices[2]->x - groupCenterOfMass.x();
	float y3 = vertices[2]->y - groupCenterOfMass.y();
	float z3 = vertices[2]->z - groupCenterOfMass.z();
	float x4 = vertices[3]->x - groupCenterOfMass.x();
	float y4 = vertices[3]->y - groupCenterOfMass.y();
	float z4 = vertices[3]->z - groupCenterOfMass.z();

	// ?寶懱??嶼嬮? A
	Eigen::Matrix4d A;
	A << x1, y1, z1, 1,
		x2, y2, z2, 1,
		x3, y3, z3, 1,
		x4, y4, z4, 1;

	// ?嶼巐柺懱揑懱?
	float V = std::abs(A.determinant() / 6);

	// 掕? mbeta, mgamma, mdelta 嬮?
	Eigen::Matrix3f mbeta1, mbeta2, mbeta3, mbeta4, mgamma1, mgamma2, mgamma3, mgamma4, mdelta1, mdelta2, mdelta3, mdelta4;


	mbeta1 << 1, y2, z2, 1, y3, z3, 1, y4, z4;
	mbeta2 << 1, y1, z1, 1, y3, z3, 1, y4, z4;
	mbeta3 << 1, y1, z1, 1, y2, z2, 1, y4, z4;
	mbeta4 << 1, y1, z1, 1, y2, z2, 1, y3, z3;

	mgamma1 << 1, x2, z2, 1, x3, z3, 1, x4, z4;
	mgamma2 << 1, x1, z1, 1, x3, z3, 1, x4, z4;
	mgamma3 << 1, x1, z1, 1, x2, z2, 1, x4, z4;
	mgamma4 << 1, x1, z1, 1, x2, z2, 1, x3, z3;

	mdelta1 << 1, x2, y2, 1, x3, y3, 1, x4, y4;
	mdelta2 << 1, x1, y1, 1, x3, y3, 1, x4, y4;
	mdelta3 << 1, x1, y1, 1, x2, y2, 1, x4, y4;
	mdelta4 << 1, x1, y1, 1, x2, y2, 1, x3, y3;

	// beta gamma delta
	float beta1 = -mbeta1.determinant();
	float beta2 = mbeta2.determinant();
	float beta3 = -mbeta3.determinant();
	float beta4 = mbeta4.determinant();

	float gamma1 = mgamma1.determinant();
	float gamma2 = -mgamma2.determinant();
	float gamma3 = mgamma3.determinant();
	float gamma4 = -mgamma4.determinant();

	float delta1 = -mdelta1.determinant();
	float delta2 = mdelta2.determinant();
	float delta3 = -mdelta3.determinant();
	float delta4 = mdelta4.determinant();

	//B matrix
	Eigen::MatrixXf B(6, 12);

	B << beta1, 0, 0, beta2, 0, 0, beta3, 0, 0, beta4, 0, 0,
		0, gamma1, 0, 0, gamma2, 0, 0, gamma3, 0, 0, gamma4, 0,
		0, 0, delta1, 0, 0, delta2, 0, 0, delta3, 0, 0, delta4,
		gamma1, beta1, 0, gamma2, beta2, 0, gamma3, beta3, 0, gamma4, beta4, 0,
		0, delta1, gamma1, 0, delta2, gamma2, 0, delta3, gamma3, 0, delta4, gamma4,
		delta1, 0, beta1, delta2, 0, beta2, delta3, 0, beta3, delta4, 0, beta4;

	B /= (6 * V);

	// Orthotropic linear elasticity in global (x,y,z).
	//
	// Previous implementation used geometric-mean coupling terms (sqrt(Ei*Ej)) for both
	// normal coupling and shear, which makes a test "drag along X vs Y" not isolate
	// Ex/Ey well (increasing Ex also boosts multiple shear modes). That can invalidate
	// the paper statement "Ex > Ey => stiffer along X".
	//
	// Here we build a symmetric compliance matrix S (Voigt order: xx,yy,zz,xy,yz,zx),
	// enforce reciprocity via nu21=nu12*E2/E1 etc, choose shear moduli based on the
	// softer axis (min(Ei,Ej)), then invert to get stiffness D.
	const float eps = 1e-12f;
	E1 = std::max(E1, eps);
	E2 = std::max(E2, eps);
	E3 = std::max(E3, eps);
	nu = std::min(0.49f, std::max(0.0f, nu));
	nu = clampNuForStableCompliance(E1, E2, E3, nu);

	const float nu12 = nu;
	const float nu13 = nu;
	const float nu23 = nu;
	const float nu21 = nu12 * (E2 / E1);
	const float nu31 = nu13 * (E3 / E1);
	const float nu32 = nu23 * (E3 / E2);

	const float inv2pnu = 1.0f / (2.0f * (1.0f + nu));
	const float G12 = std::max(std::min(E1, E2) * inv2pnu, eps); // xy
	const float G23 = std::max(std::min(E2, E3) * inv2pnu, eps); // yz
	const float G31 = std::max(std::min(E3, E1) * inv2pnu, eps); // zx

	Eigen::Matrix<float, 6, 6> S = Eigen::Matrix<float, 6, 6>::Zero();
	S(0, 0) = 1.0f / E1;
	S(1, 1) = 1.0f / E2;
	S(2, 2) = 1.0f / E3;

	// Symmetric normal coupling (nu12/E1 == nu21/E2, etc).
	S(0, 1) = -nu12 / E1;
	S(1, 0) = -nu21 / E2;
	S(0, 2) = -nu13 / E1;
	S(2, 0) = -nu31 / E3;
	S(1, 2) = -nu23 / E2;
	S(2, 1) = -nu32 / E3;

	S(3, 3) = 1.0f / G12;
	S(4, 4) = 1.0f / G23;
	S(5, 5) = 1.0f / G31;

	Eigen::Matrix<float, 6, 6> Dm = S.inverse();
	if (!Dm.allFinite()) {
		Eigen::Matrix<float, 6, 6> Sreg = S;
		Sreg.diagonal().array() += 1e-6f;
		Dm = Sreg.inverse();
	}
	Dm = projectSPD(Dm);

	Eigen::MatrixXf D = Dm;

	// element K
	Eigen::MatrixXf k = V * (B.transpose() * D * B);
	k = 0.5f * (k + k.transpose());

	elementK = k;
	return k;
}
Eigen::MatrixXf Tetrahedron::createElementK(float E, float nu, const Eigen::Vector3f& groupCenterOfMass) {

	float x1 = vertices[0]->x - groupCenterOfMass.x();
	float y1 = vertices[0]->y - groupCenterOfMass.y();
	float z1 = vertices[0]->z - groupCenterOfMass.z();
	float x2 = vertices[1]->x - groupCenterOfMass.x();
	float y2 = vertices[1]->y - groupCenterOfMass.y();
	float z2 = vertices[1]->z - groupCenterOfMass.z();
	float x3 = vertices[2]->x - groupCenterOfMass.x();
	float y3 = vertices[2]->y - groupCenterOfMass.y();
	float z3 = vertices[2]->z - groupCenterOfMass.z();
	float x4 = vertices[3]->x - groupCenterOfMass.x();
	float y4 = vertices[3]->y - groupCenterOfMass.y();
	float z4 = vertices[3]->z - groupCenterOfMass.z();


	Eigen::Matrix4d A;
	A << x1, y1, z1, 1,
		x2, y2, z2, 1,
		x3, y3, z3, 1,
		x4, y4, z4, 1;

	
	float V = std::abs(A.determinant() / 6);

	
	Eigen::Matrix3f mbeta1, mbeta2, mbeta3, mbeta4, mgamma1, mgamma2, mgamma3, mgamma4, mdelta1, mdelta2, mdelta3, mdelta4;


	mbeta1 << 1, y2, z2, 1, y3, z3, 1, y4, z4;
	mbeta2 << 1, y1, z1, 1, y3, z3, 1, y4, z4;
	mbeta3 << 1, y1, z1, 1, y2, z2, 1, y4, z4;
	mbeta4 << 1, y1, z1, 1, y2, z2, 1, y3, z3;

	mgamma1 << 1, x2, z2, 1, x3, z3, 1, x4, z4;
	mgamma2 << 1, x1, z1, 1, x3, z3, 1, x4, z4;
	mgamma3 << 1, x1, z1, 1, x2, z2, 1, x4, z4;
	mgamma4 << 1, x1, z1, 1, x2, z2, 1, x3, z3;

	mdelta1 << 1, x2, y2, 1, x3, y3, 1, x4, y4;
	mdelta2 << 1, x1, y1, 1, x3, y3, 1, x4, y4;
	mdelta3 << 1, x1, y1, 1, x2, y2, 1, x4, y4;
	mdelta4 << 1, x1, y1, 1, x2, y2, 1, x3, y3;

	
	float beta1 = -mbeta1.determinant();
	float beta2 = mbeta2.determinant();
	float beta3 = -mbeta3.determinant();
	float beta4 = mbeta4.determinant();

	float gamma1 = mgamma1.determinant();
	float gamma2 = -mgamma2.determinant();
	float gamma3 = mgamma3.determinant();
	float gamma4 = -mgamma4.determinant();

	float delta1 = -mdelta1.determinant();
	float delta2 = mdelta2.determinant();
	float delta3 = -mdelta3.determinant();
	float delta4 = mdelta4.determinant();

	
	Eigen::MatrixXf B(6, 12);

	B << beta1, 0, 0, beta2, 0, 0, beta3, 0, 0, beta4, 0, 0,
		0, gamma1, 0, 0, gamma2, 0, 0, gamma3, 0, 0, gamma4, 0,
		0, 0, delta1, 0, 0, delta2, 0, 0, delta3, 0, 0, delta4,
		gamma1, beta1, 0, gamma2, beta2, 0, gamma3, beta3, 0, gamma4, beta4, 0,
		0, delta1, gamma1, 0, delta2, gamma2, 0, delta3, gamma3, 0, delta4, gamma4,
		delta1, 0, beta1, delta2, 0, beta2, delta3, 0, beta3, delta4, 0, beta4;

	B /= (6 * V);

	
	Eigen::MatrixXf D = Eigen::MatrixXf::Zero(6, 6);

	D << 1 - nu, nu, nu, 0, 0, 0,
		nu, 1 - nu, nu, 0, 0, 0,
		nu, nu, 1 - nu, 0, 0, 0,
		0, 0, 0, (1 - 2 * nu) / 2, 0, 0,
		0, 0, 0, 0, (1 - 2 * nu) / 2, 0,
		0, 0, 0, 0, 0, (1 - 2 * nu) / 2;

	D *= (E / ((1 + nu) * (1 - 2 * nu)));

	
	Eigen::MatrixXf k = V * (B.transpose() * D * B);

	elementK = k;
	return k;
}

Eigen::MatrixXf Tetrahedron::createElementKFEM(float E, float nu) {
	
	float x1 = vertices[0]->x;
	float y1 = vertices[0]->y;
	float z1 = vertices[0]->z;
	float x2 = vertices[1]->x;
	float y2 = vertices[1]->y;
	float z2 = vertices[1]->z;
	float x3 = vertices[2]->x;
	float y3 = vertices[2]->y;
	float z3 = vertices[2]->z;
	float x4 = vertices[3]->x;
	float y4 = vertices[3]->y;
	float z4 = vertices[3]->z;


	Eigen::Matrix4d A;
	A << x1, y1, z1, 1,
		x2, y2, z2, 1,
		x3, y3, z3, 1,
		x4, y4, z4, 1;

	
	float V = std::abs(A.determinant() / 6);

	
	Eigen::Matrix3f mbeta1, mbeta2, mbeta3, mbeta4, mgamma1, mgamma2, mgamma3, mgamma4, mdelta1, mdelta2, mdelta3, mdelta4;


	mbeta1 << 1, y2, z2, 1, y3, z3, 1, y4, z4;
	mbeta2 << 1, y1, z1, 1, y3, z3, 1, y4, z4;
	mbeta3 << 1, y1, z1, 1, y2, z2, 1, y4, z4;
	mbeta4 << 1, y1, z1, 1, y2, z2, 1, y3, z3;

	mgamma1 << 1, x2, z2, 1, x3, z3, 1, x4, z4;
	mgamma2 << 1, x1, z1, 1, x3, z3, 1, x4, z4;
	mgamma3 << 1, x1, z1, 1, x2, z2, 1, x4, z4;
	mgamma4 << 1, x1, z1, 1, x2, z2, 1, x3, z3;

	mdelta1 << 1, x2, y2, 1, x3, y3, 1, x4, y4;
	mdelta2 << 1, x1, y1, 1, x3, y3, 1, x4, y4;
	mdelta3 << 1, x1, y1, 1, x2, y2, 1, x4, y4;
	mdelta4 << 1, x1, y1, 1, x2, y2, 1, x3, y3;

	
	float beta1 = -mbeta1.determinant();
	float beta2 = mbeta2.determinant();
	float beta3 = -mbeta3.determinant();
	float beta4 = mbeta4.determinant();

	float gamma1 = mgamma1.determinant();
	float gamma2 = -mgamma2.determinant();
	float gamma3 = mgamma3.determinant();
	float gamma4 = -mgamma4.determinant();

	float delta1 = -mdelta1.determinant();
	float delta2 = mdelta2.determinant();
	float delta3 = -mdelta3.determinant();
	float delta4 = mdelta4.determinant();

	
	Eigen::MatrixXf B(6, 12);

	B << beta1, 0, 0, beta2, 0, 0, beta3, 0, 0, beta4, 0, 0,
		0, gamma1, 0, 0, gamma2, 0, 0, gamma3, 0, 0, gamma4, 0,
		0, 0, delta1, 0, 0, delta2, 0, 0, delta3, 0, 0, delta4,
		gamma1, beta1, 0, gamma2, beta2, 0, gamma3, beta3, 0, gamma4, beta4, 0,
		0, delta1, gamma1, 0, delta2, gamma2, 0, delta3, gamma3, 0, delta4, gamma4,
		delta1, 0, beta1, delta2, 0, beta2, delta3, 0, beta3, delta4, 0, beta4;

	B /= (6 * V);

	
	Eigen::MatrixXf D = Eigen::MatrixXf::Zero(6, 6);

	D << 1 - nu, nu, nu, 0, 0, 0,
		nu, 1 - nu, nu, 0, 0, 0,
		nu, nu, 1 - nu, 0, 0, 0,
		0, 0, 0, (1 - 2 * nu) / 2, 0, 0,
		0, 0, 0, 0, (1 - 2 * nu) / 2, 0,
		0, 0, 0, 0, 0, (1 - 2 * nu) / 2;

	D *= (E / ((1 + nu) * (1 - 2 * nu)));

	
	Eigen::MatrixXf k = V * (B.transpose() * D * B);

	elementKFEM = k;
	return k;
}
float Tetrahedron::calMassTetra(float den) {

	//float volume;
	Eigen::Vector3f AB(vertices[1]->x - vertices[0]->x, vertices[1]->y - vertices[0]->y, vertices[1]->z - vertices[0]->z);
	Eigen::Vector3f AC(vertices[2]->x - vertices[0]->x, vertices[2]->y - vertices[0]->y, vertices[2]->z - vertices[0]->z);
	Eigen::Vector3f AD(vertices[3]->x - vertices[0]->x, vertices[3]->y - vertices[0]->y, vertices[3]->z - vertices[0]->z);

	// Calculate volume using the formula
	volumeTetra = (AB.cross(AC)).dot(AD) / 6.0f;
	volumeTetra = std::abs(volumeTetra);
	massTetra = volumeTetra * den;
	return massTetra;


}
float Tetrahedron::calVolumeTetra() {

	//float volume;
	Eigen::Vector3f AB(vertices[1]->x - vertices[0]->x, vertices[1]->y - vertices[0]->y, vertices[1]->z - vertices[0]->z);
	Eigen::Vector3f AC(vertices[2]->x - vertices[0]->x, vertices[2]->y - vertices[0]->y, vertices[2]->z - vertices[0]->z);
	Eigen::Vector3f AD(vertices[3]->x - vertices[0]->x, vertices[3]->y - vertices[0]->y, vertices[3]->z - vertices[0]->z);

	// Calculate volume using the formula
	volumeTetra = (AB.cross(AC)).dot(AD) / 6.0f;
	volumeTetra = std::abs(volumeTetra);
	return volumeTetra;
}
