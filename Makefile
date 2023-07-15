format:
	poetry run black skinnyglms/
	poetry run isort skinnyglms/
	
	poetry run black tests/
	poetry run isort tests/
	
	poetry run black examples/
	poetry run isort examples/
lint: 
	poetry run ruff check skinnyglms/. --fix
	poetry run ruff check tests/. --fix

test:
	poetry run pytest tests/.